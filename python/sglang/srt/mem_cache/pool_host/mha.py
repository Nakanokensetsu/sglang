from __future__ import annotations

import logging
import threading
from typing import Sequence

import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_hicache_jit_kernel,
    can_use_write_back_jit_kernel,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_staged_lf_pf as jit_transfer_hicache_all_layer_staged_lf_pf,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer as jit_transfer_hicache_one_layer,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKOnlyPool, MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    HostKVCache,
    host_memory_budget_bytes,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)
from sglang.srt.utils import is_cuda, is_hip, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer,
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_lf_pf,
        transfer_kv_all_layer_lf_ph,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
        transfer_kv_per_layer_pf_lf,
        transfer_kv_per_layer_ph_lf,
    )
if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange

logger = logging.getLogger(__name__)


class MHATokenToKVPoolHost(HostKVCache):
    device_pool: MHATokenToKVPool | None = None
    mtp_draft_device_pools: tuple[MHATokenToKVPool, ...] = ()

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        mtp_draft_device_pools: Sequence[MHATokenToKVPool] = (),
        pool_label: str = "kv",
    ):
        self.mtp_draft_device_pools = tuple(mtp_draft_device_pools)
        self.target_layer_num = device_pool.layer_num
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            pool_label=pool_label,
        )
        self.element_dim = self._compute_element_dim()
        # The JIT HiCache kernels also build with hipcc (ROCm): the PTX-only
        # helpers in hicache.cuh are guarded by USE_ROCM and the staged
        # write-back kernel has a ROCm path, so enable them on HIP too. This
        # keeps the ROCm write-back path consistent with CUDA.
        self.can_use_jit = (_is_cuda or _is_hip) and can_use_hicache_jit_kernel(
            element_size=self.element_dim * self.dtype.itemsize
        )

        if self.layout == "page_first":
            # Transpose [page, layer, ...] -> [layer, page, ...] to get per-layer views
            # This swaps strides without copying data
            k_transposed = self.k_buffer.transpose(0, 1)
            v_transposed = self.v_buffer.transpose(0, 1)
            self.k_data_refs = [k_transposed[i] for i in range(self.layer_num)]
            self.v_data_refs = [v_transposed[i] for i in range(self.layer_num)]
        else:
            self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
            self.v_data_refs = [self.v_buffer[i] for i in range(self.layer_num)]
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.mtp_draft_device_pools:
            device_pools = (self.device_pool, *self.mtp_draft_device_pools)
            if not _is_npu:
                self.packed_device_k_data_ptrs = torch.cat(
                    [pool.k_data_ptrs for pool in device_pools]
                )
                self.packed_device_v_data_ptrs = torch.cat(
                    [pool.v_data_ptrs for pool in device_pools]
                )
            else:
                self.packed_device_k_data_ptrs = None
                self.packed_device_v_data_ptrs = None
            self.packed_device_k_buffers = [
                buffer for pool in device_pools for buffer in pool.k_buffer
            ]
            self.packed_device_v_buffers = [
                buffer for pool in device_pools for buffer in pool.v_buffer
            ]
            self.packed_device_kv_buffers = (
                self.packed_device_k_buffers + self.packed_device_v_buffers
            )
        self.host_kv_data_refs = self.k_data_refs + self.v_data_refs
        self._init_write_back_staging_buffers()

    def _compute_element_dim(self) -> int:
        """1トークン・1層・K(または V)あたりの要素数。

        ホスト本体バッファの1トークン幅(``token_stride_size``)および
        デバイス側の1行のバイト数と**必ず一致**しなければならない。転送カーネルは
        この値を ``element_dim`` / ``element_size`` として受け取り、src/dst の
        両方へ同じ幅を適用する(``kv_cache_src_stride_bytes`` にホスト側の
        ``token_stride_size`` を渡しているのがその証拠)。

        2026-09-20 自前追加: デバイスの ``head_dim`` がそのままバイト数に
        ならない量子化プール(int2)のためにサブクラスが差し替えられるよう
        メソッドへ切り出した。
        """
        return self.device_pool.head_num * self.device_pool.head_dim

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_ksize_per_token(self):
        return self.get_size_per_token() // 2

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_head":
            dims = (
                2,
                self.page_num,
                self.head_num,
                self.page_size,
                self.layer_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=(
                self.page_size * self.layout_dim
                if self.layout in ("page_first", "page_first_direct")
                else None
            ),
        )
        return buffer

    def _init_write_back_staging_buffers(self):
        self.staging_page_capacity = 0
        self.staging_token_capacity = 0
        self.staging_k_buffer = None
        self.staging_v_buffer = None
        self.can_use_write_back_jit = False
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        # The staged write-back JIT kernel builds with hipcc and has a ROCm
        # path, so enable it on HIP too (consistent with the CUDA path).
        self.can_use_write_back_jit = (
            _is_cuda or _is_hip
        ) and can_use_write_back_jit_kernel(
            element_size=self.element_dim * self.dtype.itemsize,
        )
        if not self.can_use_write_back_jit:
            return

        self.staging_page_capacity = min(self.page_num, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_token_capacity = self.staging_page_capacity * self.page_size
        self.staging_k_buffer = torch.empty(
            (
                self.staging_token_capacity,
                self.layer_num,
                self.head_num,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device_pool.device,
        )
        self.staging_v_buffer = torch.empty_like(self.staging_k_buffer)

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if self.device_pool is not None:
            if not is_draft and not self._is_device_layer_owned(device_pool, layer_id):
                return
            # MTP draft layers do not participate in CP layer sharding.
            host_layer_id = layer_id if is_draft else self._host_layer_index(layer_id)
            device_layer_id = 0 if is_draft else layer_id
        else:
            host_layer_id = device_layer_id = layer_id

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[device_layer_id],
                        v_cache_dst=device_pool.v_buffer[device_layer_id],
                        k_cache_src=self.k_buffer[host_layer_id],
                        v_cache_src=self.v_buffer[host_layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer(
                        src_k=self.k_buffer[host_layer_id],
                        dst_k=device_pool.k_buffer[device_layer_id],
                        src_v=self.v_buffer[host_layer_id],
                        dst_v=device_pool.v_buffer[device_layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    # Transpose [page, layer, ...] -> [layer, page, ...] then
                    # index by layer_id to get a per-layer view with strided layout.
                    # The kernel handles different src/dst strides automatically.
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[device_layer_id],
                        v_cache_dst=device_pool.v_buffer[device_layer_id],
                        k_cache_src=self.k_data_refs[host_layer_id],
                        v_cache_src=self.v_data_refs[host_layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer_pf_lf(
                        src_k=self.k_buffer,
                        dst_k=device_pool.k_buffer[device_layer_id],
                        src_v=self.v_buffer,
                        dst_v=device_pool.v_buffer[device_layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=host_layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            elif self.layout == "page_head":
                transfer_kv_per_layer_ph_lf(
                    src_k=self.k_buffer,
                    dst_k=device_pool.k_buffer[device_layer_id],
                    src_v=self.v_buffer,
                    dst_v=device_pool.v_buffer[device_layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=host_layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[
                        self.k_buffer[host_layer_id],
                        self.v_buffer[host_layer_id],
                    ],
                    dst_layers=[
                        device_pool.k_buffer[device_layer_id],
                        device_pool.v_buffer[device_layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer, self.v_buffer],
                    dst_ptrs=[
                        device_pool.k_buffer[device_layer_id],
                        device_pool.v_buffer[device_layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=host_layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if host_layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _resolve_device_transfer_buffers(self, device_pool):
        if self.mtp_draft_device_pools:
            return (
                self.packed_device_k_data_ptrs,
                self.packed_device_v_data_ptrs,
                self.packed_device_k_buffers,
                self.packed_device_v_buffers,
            )
        return (
            device_pool.k_data_ptrs,
            device_pool.v_data_ptrs,
            device_pool.k_buffer,
            device_pool.v_buffer,
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel_ascend":
            # NPU pools use contiguous multi-layer tensors and intentionally do
            # not build the CUDA-style k_data_ptrs/v_data_ptrs arrays.
            device_kv_buffers = None
        else:
            (
                device_k_data_ptrs,
                device_v_data_ptrs,
                device_k_buffers,
                device_v_buffers,
            ) = self._resolve_device_transfer_buffers(device_pool)
            device_kv_buffers = device_k_buffers + device_v_buffers
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_k_data_ptrs,
                        v_ptr_src=device_v_data_ptrs,
                        indices_src=device_indices,
                        kv_cache_dst_stride_bytes=self.token_stride_size,
                        kv_cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.element_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer(
                        src_k_layers=device_k_data_ptrs,
                        dst_k_layers=self.k_data_ptrs,
                        src_v_layers=device_v_data_ptrs,
                        dst_v_layers=self.v_data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_first":
                if self.can_use_write_back_jit:
                    jit_transfer_hicache_all_layer_staged_lf_pf(
                        k_ptr_src=device_k_data_ptrs,
                        v_ptr_src=device_v_data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        staging_k=self.staging_k_buffer,
                        staging_v=self.staging_v_buffer,
                        dst_k=self.k_buffer,
                        dst_v=self.v_buffer,
                        page_size=self.page_size,
                    )
                else:
                    transfer_kv_all_layer_lf_pf(
                        src_k_layers=device_k_data_ptrs,
                        dst_k=self.k_buffer,
                        src_v_layers=device_v_data_ptrs,
                        dst_v=self.v_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_head":
                transfer_kv_all_layer_lf_ph(
                    src_k_layers=device_k_data_ptrs,
                    dst_k=self.k_buffer,
                    src_v_layers=device_v_data_ptrs,
                    dst_v=self.v_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_kv_buffers,
                    dst_layers=self.host_kv_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_kv_buffers,
                    dst_ptrs=[self.k_buffer, self.v_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :, :]
        elif self.layout in ["page_first_direct", "page_head"]:
            real_index = index // self.page_size
            data_page = self.kv_buffer[:, real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, :, index : index + self.page_size, :, :] = (
                data_page.reshape(
                    2,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
            )
        elif self.layout == "page_first":
            self.kv_buffer[:, index : index + self.page_size, :, :, :] = (
                data_page.reshape(
                    2, self.page_size, self.layer_num, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.layer_num, self.page_size, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_head":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.head_num, self.page_size, self.layer_num, self.head_dim
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        """
        get meta data for zero copy of heterogeneous ranks' KVCache
        """
        assert self.layout == "page_head"
        assert len(indices) % self.page_size == 0
        assert self.head_num % split_factor == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        for index in range(0, len(indices), self.page_size):
            for head_id in range(0, self.head_num, self.head_num // split_factor):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                    + head_id
                    * self.page_size
                    * self.layer_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
            // split_factor
        )
        element_size_list = [element_size] * len(ptr_list)
        return ptr_list, element_size_list

    def get_page_buffer_meta(self, indices):
        """
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index]
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                        + layer_id
                        * self.size
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                    )
                    v_ptr = k_ptr + v_offset
                    ptr_list.append(k_ptr)
                    ptr_list.append(v_ptr)
            element_size = (
                self.dtype.itemsize * self.page_size * self.head_num * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct", "page_head"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.head_num
                * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """Return True if per-page strides are multiples of *page_size_bytes*.

        When O_DIRECT is used with any file-based NIXL backend, every data pointer
        passed to the kernel must be page-aligned.  In zero-copy mode the
        pointer for KV page ``p`` is:

            base_ptr + p * page_size * layer_num * head_num * head_dim * itemsize

        For this to be page-aligned (given a page-aligned ``base_ptr``) the per-page
        stride must itself be a multiple of the OS page size.
        """
        if self.layout not in ("page_first", "page_first_direct", "page_head"):
            return False
        stride = (
            self.page_size
            * self.layer_num
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        base_aligned = self.kv_buffer.data_ptr() % page_size_bytes == 0
        return base_aligned and stride % page_size_bytes == 0


class MHATokenToKOnlyPoolHost(HostKVCache):
    """Host pool for MiniMax sparse index-K buffers (no index V)."""

    device_pool: MHATokenToKOnlyPool

    def __init__(
        self,
        device_pool: MHATokenToKOnlyPool,
        anchor_host: MHATokenToKVPoolHost,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = anchor_host.page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        self.head_num = device_pool.head_num
        self.head_dim = device_pool.head_dim
        self.layer_num = device_pool.layer_num
        self.element_dim = self.head_num * self.head_dim
        self.token_stride_size = self.element_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        self.size = anchor_host.size
        self.page_num = anchor_host.page_num
        self.size_per_token = self.get_size_per_token()

        requested_bytes = self.size * self.size_per_token
        available_bytes = host_memory_budget_bytes()
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for MiniMax index-K hierarchical cache. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )
        logger.info(
            "Allocating %.2f GB host memory for MiniMax sparse index-K (layout=%s).",
            requested_bytes / 1e9,
            layout,
        )

        self.init_kv_buffer()
        self.lock = threading.RLock()
        self.clear()

        self.can_use_jit = _is_cuda and can_use_hicache_jit_kernel(
            element_size=self.token_stride_size
        )
        self.k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.k_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "page_first":
            transposed = self.k_buffer.transpose(0, 1)
            self.k_data_refs = [transposed[i] for i in range(self.layer_num)]
        elif self.layout == "layer_first":
            self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
        else:
            self.k_data_refs = []
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.k_buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=(
                self.page_size * self.layout_dim
                if self.layout in ("page_first", "page_first_direct")
                else None
            ),
        )

    def get_hybrid_pool_buffer(self):
        return [self.k_buffer]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.k_buffer[layer_id],
                        cache_src=self.k_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer_mla(
                        src=self.k_buffer[layer_id],
                        dst=device_pool.k_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.k_buffer[layer_id],
                        cache_src=self.k_data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer_mla_pf_lf(
                        src=self.k_buffer,
                        dst=device_pool.k_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id]],
                    dst_layers=[device_pool.k_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer],
                    dst_ptrs=[device_pool.k_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    for layer_id in range(self.layer_num):
                        jit_transfer_hicache_one_layer_mla(
                            cache_dst=self.k_buffer[layer_id],
                            cache_src=device_pool.k_buffer[layer_id],
                            indices_dst=host_indices,
                            indices_src=device_indices,
                            element_dim=self.element_dim,
                        )
                else:
                    for layer_id in range(self.layer_num):
                        transfer_kv_per_layer_mla(
                            src=device_pool.k_buffer[layer_id],
                            dst=self.k_buffer[layer_id],
                            src_indices=device_indices,
                            dst_indices=host_indices,
                            item_size=self.token_stride_size,
                        )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.k_data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=self.k_device_ptrs,
                        indices_src=device_indices,
                        cache_dst_stride_bytes=self.layout_dim,
                        cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.element_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=self.k_device_ptrs,
                        dst=self.k_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer,
                    dst_layers=self.k_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.k_buffer,
                    dst_ptrs=[self.k_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.k_buffer[:, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.k_buffer[index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.k_buffer[real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            return data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.k_buffer[:, index : index + self.page_size, :, :] = data_page.reshape(
                self.layer_num, self.page_size, self.head_num, self.head_dim
            )
        elif self.layout == "page_first":
            self.k_buffer[index : index + self.page_size, :, :, :] = data_page.reshape(
                self.page_size, self.layer_num, self.head_num, self.head_dim
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.k_buffer[real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                1, self.layer_num, self.page_size, self.head_num, self.head_dim
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """Meta data for zero-copy storage I/O."""
        assert len(indices) % self.page_size == 0
        if self.layout not in ["page_first", "page_first_direct"]:
            raise ValueError(f"Unsupported layout: {self.layout}")

        ptr_list = []
        k_buffer_data_ptr = self.k_buffer.data_ptr()
        indices = indices.tolist()
        for index in range(0, len(indices), self.page_size):
            k_ptr = (
                k_buffer_data_ptr
                + indices[index]
                * self.layer_num
                * self.head_num
                * self.head_dim
                * self.dtype.itemsize
            )
            ptr_list.append(k_ptr)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        element_size_list = [element_size] * len(ptr_list)
        return ptr_list, element_size_list


class Int2MHATokenToKVPoolHost(MHATokenToKVPoolHost):
    """Host KV pool for the OSCAR INT2 device pool (2026-09-20 自前追加).

    素の ``MHATokenToKVPoolHost`` は1トークンを
    ``head_num * head_dim * dtype.itemsize`` バイトの連続領域として扱う。
    INT2 プールはそれが2点で成り立たない:

    1. packed codes は ``[slots, head_num, head_dim // 4]`` uint8。
       1ヘッドあたり ``head_dim`` ではなく ``head_dim // 4`` バイトしかない。
    2. 量子化の scale/zero は ``k_scales_zeros`` という**別テンソル**
       (``[slots, head_num, 2 * num_groups]``, ``scale_dtype``)。これを運ばないと
       ホストから戻したスロットは**前の占有者の scale/zero で解釈され、静かに壊れる**。

    【2026-09-20 修正】最初の実装は「1トークン = packed + sz の連続領域」とみなして
    ``head_dim`` を実効バイト数(packed + sz)へ差し替えていた。これが誤りだった:
    sz は結局**別バッファ**に置いたので、ホスト本体バッファだけが幅広になり、
    転送カーネルはデバイス側の1行(= packed のみ)を超えて読み書きしていた。
    さらに親の ``element_dim`` は**デバイスの** ``head_dim`` から算出されるため、
    そこだけ4倍幅のまま残っていた。**転送に関わる3つの幅が三者三様**という状態で、
    症状は毎回「最初の長文リクエストで CUDA illegal memory access」になる。

    現在の取り決め:
      * ホスト本体バッファは **packed codes だけ**を持つ。1ヘッド ``head_dim // 4``
        バイト。これでホストの ``token_stride_size`` とデバイスの行バイト数が一致する。
      * scales/zeros は ``host_k_sz`` / ``host_v_sz`` に別建てで持ち、packed と
        同じ ``host_indices`` / ``device_indices`` で別途運ぶ。
      * ``get_size_per_token`` が返すのは **packed + sz を足した**バイト数。
        ホストRAM から取れるスロット数 ``self.size`` はこれで決まるので、
        sz を数えないと確保量を過小申告することになる。
      * 3つの幅は ``_assert_transfer_geometry`` が**起動時に**突き合わせる。
        食い違ったまま起動させない。
    """

    def __init__(self, device_pool, *args, **kwargs):
        assert getattr(device_pool, "dtype", None) == "int2", (
            "Int2MHATokenToKVPoolHost requires an int2 device pool, got "
            f"{getattr(device_pool, 'dtype', None)!r}"
        )
        # packed: head_dim//4 バイト/ヘッド (uint8)
        self._packed_bytes_per_head_k = device_pool.head_dim // 4
        self._packed_bytes_per_head_v = device_pool.v_head_dim // 4
        # scales/zeros: 2 * num_groups 要素/ヘッド
        sz_itemsize = torch.empty(0, dtype=device_pool.scale_dtype).element_size()
        self._sz_bytes_per_head_k = 2 * device_pool.k_num_scale_groups * sz_itemsize
        self._sz_bytes_per_head_v = 2 * device_pool.v_num_scale_groups * sz_itemsize
        if (
            self._packed_bytes_per_head_k != self._packed_bytes_per_head_v
            or self._sz_bytes_per_head_k != self._sz_bytes_per_head_v
        ):
            # 親は K と V を同じ dims の1本の ``kv_buffer`` に確保する。非対称は
            # ``AsymmetricMHATokenToKVPoolHost`` 相当の作り直しが要る。
            raise NotImplementedError(
                "Int2 HiCache host pool requires symmetric K/V geometry: "
                f"k packed={self._packed_bytes_per_head_k}B "
                f"sz={self._sz_bytes_per_head_k}B vs "
                f"v packed={self._packed_bytes_per_head_v}B "
                f"sz={self._sz_bytes_per_head_v}B."
            )
        super().__init__(device_pool, *args, **kwargs)
        self._assert_transfer_geometry()

    def _compute_element_dim(self) -> int:
        # 親はデバイスの head_dim(例: 256)から算出するが、int2 のデバイス行は
        # head_dim//4 バイトしかない。ホスト本体バッファの幅に合わせる。
        return self.head_num * self._packed_bytes_per_head_k

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        # ホスト本体バッファは packed codes のみ。dtype は uint8(itemsize=1)なので、
        # head_dim にバイト数をそのまま入れれば親の init_kv_buffer /
        # token_stride_size がデバイス側と同じ幅を導く。
        self.head_dim = self._packed_bytes_per_head_k
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)
        # 容量計算(self.size の決定)には sz ぶんも含める。ホストRAMは両方食う。
        k_bytes = self.head_num * (
            self._packed_bytes_per_head_k + self._sz_bytes_per_head_k
        )
        v_bytes = self.head_num * (
            self._packed_bytes_per_head_v + self._sz_bytes_per_head_v
        )
        return (k_bytes + v_bytes) * self.layer_num

    def get_ksize_per_token(self):
        k_bytes = self.head_num * (
            self._packed_bytes_per_head_k + self._sz_bytes_per_head_k
        )
        return k_bytes * self.layer_num

    def init_kv_buffer(self):
        """packed 用の本体バッファ。``head_dim`` は packed バイト数に置き換え済みなので、
        親の実装がそのまま正しいサイズを確保する。
        scales/zeros は ``_init_int2_sz_buffer`` が別に確保する。"""
        buf = super().init_kv_buffer()
        self._init_int2_sz_buffer()
        return buf

    def _init_int2_sz_buffer(self):
        """scales/zeros 用のホストバッファ(K/V 各層)。

        packed と同じ ``self.size`` スロットを持ち、1スロットあたり
        ``head_num * 2 * num_groups`` 要素 (``scale_dtype``)。
        packed と同じインデックス空間を使うので、転送は同じ host_indices で済む。
        """
        dp = self.device_pool
        alloc_func = ALLOC_MEMORY_FUNCS[dp.device]
        self.sz_dtype = dp.scale_dtype
        k_dims = (self.layer_num, self.size, self.head_num, 2 * dp.k_num_scale_groups)
        v_dims = (self.layer_num, self.size, self.head_num, 2 * dp.v_num_scale_groups)
        self.host_k_sz = alloc_func(
            k_dims,
            dtype=self.sz_dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=None,
        )
        self.host_v_sz = alloc_func(
            v_dims,
            dtype=self.sz_dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=None,
        )

    def _assert_transfer_geometry(self):
        """転送に関わる幅を起動時に突き合わせる。

        ここが食い違うと症状は必ず「最初の長文リクエストで CUDA illegal memory
        access」になり、原因の切り分けに非常に時間がかかる(2026-09-20 に3回踏んだ)。
        起動時1回きりなので常に実行する。
        """
        dp = self.device_pool

        # MTP draft プールは packed を packed_device_*_ptrs 経由でまとめて運ぶが、
        # sz 側にその配線が無い。黙って sz を落とすと静かに壊れるので拒否する。
        if self.mtp_draft_device_pools:
            raise NotImplementedError(
                "int2 HiCache does not support MTP draft KV pools yet: the "
                "scales/zeros transfer has no packed-draft path."
            )

        def _row_bytes(t: torch.Tensor) -> int:
            # 1スロット(= 1トークン)ぶんのバイト数。
            return t[0].numel() * t.element_size()

        dev_k_row = _row_bytes(dp.k_buffer[0])
        dev_v_row = _row_bytes(dp.v_buffer[0])
        host_row = self.token_stride_size
        element_bytes = self.element_dim * self.dtype.itemsize

        detail = (
            f"host token_stride={host_row}B, element_dim*itemsize={element_bytes}B, "
            f"device k row={dev_k_row}B, device v row={dev_v_row}B "
            f"(head_num={self.head_num}, device head_dim={dp.head_dim}, "
            f"packed/head={self._packed_bytes_per_head_k}B, "
            f"sz/head={self._sz_bytes_per_head_k}B)"
        )
        assert dev_k_row == dev_v_row, f"int2 device K/V rows differ: {detail}"
        assert host_row == dev_k_row, (
            "int2 HiCache host token stride does not match the device row; "
            f"transfers would run past the device buffer. {detail}"
        )
        assert element_bytes == dev_k_row, (
            "int2 HiCache element_dim does not match the device row; "
            f"the JIT kernel would copy the wrong width. {detail}"
        )
        assert self.dtype == dp.store_dtype, (
            f"int2 HiCache host dtype {self.dtype} != device store_dtype "
            f"{dp.store_dtype}"
        )

        # scales/zeros 側。ホストとデバイスで1スロットの要素数・dtype が一致すること。
        host_k_sz_row = self.host_k_sz[0][0].numel()
        dev_k_sz_row = dp.k_scales_zeros[0][0].numel()
        host_v_sz_row = self.host_v_sz[0][0].numel()
        dev_v_sz_row = dp.v_scales_zeros[0][0].numel()
        assert host_k_sz_row == dev_k_sz_row and host_v_sz_row == dev_v_sz_row, (
            "int2 HiCache scales/zeros row mismatch: host "
            f"k={host_k_sz_row}/v={host_v_sz_row} elems, device "
            f"k={dev_k_sz_row}/v={dev_v_sz_row} elems"
        )
        assert (
            self.host_k_sz.dtype == dp.k_scales_zeros[0].dtype
            and self.host_v_sz.dtype == dp.v_scales_zeros[0].dtype
        ), (
            f"int2 HiCache scales/zeros dtype mismatch: host {self.host_k_sz.dtype}, "
            f"device {dp.k_scales_zeros[0].dtype}"
        )

        # 層数。sz バッファは host 側の layer_num 本、device は自分の layer_num 本。
        assert self.host_k_sz.shape[0] == self.layer_num
        assert len(dp.k_scales_zeros) == len(dp.k_buffer)

        # ホスト側スロット数。packed と sz が同じインデックス空間を使う前提。
        assert self.host_k_sz.shape[1] == self.size, (
            f"int2 HiCache sz host slots {self.host_k_sz.shape[1]} != "
            f"packed slots {self.size}"
        )

        logger.info(
            "Int2 HiCache host pool geometry verified: %d B/token/layer packed "
            "(K+V), %d B/token/layer scales+zeros (K+V), %d layers, %d slots.",
            2 * host_row,
            2 * self.head_num * self._sz_bytes_per_head_k,
            self.layer_num,
            self.size,
        )

    def _transfer_int2_scales_zeros(
        self, host_indices, device_indices, layer_pairs, to_host: bool
    ):
        """scales/zeros を層ごとに運ぶ。

        packed 側は親の K/V 経路がそのまま運ぶので、ここは scales/zeros だけを担当する。
        JIT カーネルは ``element_size % 128 == 0`` を要求する
        (``can_use_hicache_jit_kernel``)が、scales/zeros は1行が数十バイトしかなく
        不適格なので torch のインデックス代入で運ぶ。量は packed の 1/2 以下で
        転送1回につき1度なので、ここがボトルネックにはならない。

        ``layer_pairs`` は ``(host_layer_id, device_layer_id)`` の列。CP 層分割や
        MTP draft では両者がずれるため、呼び出し側が親と同じ対応で渡す。
        呼び出しは ``L2TransferEngine`` の転送ストリーム内で行われるので、ここで
        発行する torch の演算も同じストリームに乗る。
        """
        dp = self.device_pool
        h_idx = host_indices.to(device="cpu", dtype=torch.int64)
        d_idx = device_indices.to(device=dp.k_buffer[0].device, dtype=torch.int64)
        # インデックス範囲の検査。ここが原因なら illegal access ではなく
        # この assert で止まるので、packed 側の非同期エラーと区別できる。
        if h_idx.numel():
            h_max = int(h_idx.max())
            h_min = int(h_idx.min())
            assert 0 <= h_min and h_max < self.host_k_sz.shape[1], (
                f"int2 HiCache host index out of range: [{h_min}, {h_max}] "
                f"vs host slots {self.host_k_sz.shape[1]}"
            )
        if d_idx.numel():
            dev_slots = dp.k_scales_zeros[0].shape[0]
            d_max = int(d_idx.max())
            d_min = int(d_idx.min())
            assert 0 <= d_min and d_max < dev_slots, (
                f"int2 HiCache device index out of range: [{d_min}, {d_max}] "
                f"vs device slots {dev_slots}"
            )
        for host_layer_id, device_layer_id in layer_pairs:
            for host_buf, dev_bufs in (
                (self.host_k_sz, dp.k_scales_zeros),
                (self.host_v_sz, dp.v_scales_zeros),
            ):
                h = host_buf[host_layer_id]
                d = dev_bufs[device_layer_id]
                if to_host:
                    # D2H。CPU 側への散布は値が届いてからでないとできないので、
                    # ここは同期コピーになる(転送ストリーム上の1回だけ)。
                    h[h_idx] = d[d_idx].to("cpu")
                else:
                    d[d_idx] = h[h_idx].to(d.device)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        # 親の all-layer 経路はデバイスの全所有層を運ぶ。sz も同じ対応で運ぶ。
        if self.device_pool is not None:
            device_layer_ids = self._owned_device_layer_ids(device_pool)
            layer_pairs = [
                (self._host_layer_index(lid, device_pool), lid)
                for lid in device_layer_ids
            ]
        else:
            layer_pairs = [(lid, lid) for lid in range(self.layer_num)]
        self._transfer_int2_scales_zeros(
            host_indices, device_indices, layer_pairs, to_host=True
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        super().load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )
        # 親と同じ層対応。所有していない層は親が早期 return するのでここも合わせる。
        if self.device_pool is not None:
            if not is_draft and not self._is_device_layer_owned(device_pool, layer_id):
                return
            host_layer_id = (
                layer_id if is_draft else self._host_layer_index(layer_id, device_pool)
            )
            device_layer_id = 0 if is_draft else layer_id
        else:
            host_layer_id = device_layer_id = layer_id
        self._transfer_int2_scales_zeros(
            host_indices, device_indices, ((host_layer_id, device_layer_id),),
            to_host=False,
        )

    # --- L3 (ストレージ backend) 経路は未対応 ---------------------------------
    # 親の flat data page は本体バッファ(= packed codes)だけを平坦化する。int2 は
    # scales/zeros が別バッファなので、そのまま書き出すと復元側で前の占有者の
    # scale/zero を使うことになり、静かに壊れる。L2(ホストRAM)だけを使う分には
    # これらは呼ばれない(呼ぶのは storage backend のみ)。

    def _l3_unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "int2 HiCache does not support L3 storage backends yet: the flat "
            "data page carries only the packed codes, not the scales/zeros."
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise self._l3_unsupported()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise self._l3_unsupported()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise self._l3_unsupported()

    def get_page_buffer_meta(self, indices):
        raise self._l3_unsupported()

    def get_split_heads_page_buffer_meta(self, *args, **kwargs):
        raise self._l3_unsupported()


class AsymmetricMHATokenToKVPoolHost(MHATokenToKVPoolHost):
    """Host KV pool for MHA models whose K and V have different head dims
    (``head_dim != v_head_dim``), e.g. MiMo-V2.

    K and V are stored in two independent host buffers (``self.k_buffer`` and
    ``self.v_buffer``) instead of a single ``(2, ...)`` tensor, so each side
    keeps its native stride. The kernel transfer path dispatches K and V as
    independent single-buffer copies so each side uses its own ``item_size``.
    K/V direct transfers must be dispatched separately because the direct
    kernels derive copy sizes from each call's first tensor.
    """

    def _init_write_back_staging_buffers(self):
        self.staging_page_capacity = 0
        self.staging_token_capacity = 0
        self.staging_k_buffer = None
        self.staging_v_buffer = None
        self.can_use_write_back_jit = False
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        # K and V have different element sizes. Use the single-buffer staged
        # kernel for each side, which specializes to its native stride.
        can_use_staged_jit = (_is_cuda or _is_hip) and all(
            can_use_write_back_jit_kernel(element_size=element_size)
            for element_size in (
                self._k_token_stride_size(),
                self._v_token_stride_size(),
            )
        )
        if not can_use_staged_jit:
            return

        self.can_use_write_back_jit = True
        self.staging_page_capacity = min(self.page_num, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_token_capacity = self.staging_page_capacity * self.page_size
        self.staging_k_buffer = torch.empty(
            (
                self.staging_token_capacity,
                self.layer_num,
                self.head_num,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device_pool.device,
        )
        self.staging_v_buffer = torch.empty(
            (
                self.staging_token_capacity,
                self.layer_num,
                self.head_num,
                self.v_head_dim,
            ),
            dtype=self.dtype,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)
        self.v_head_dim = self.device_pool.v_head_dim
        return (
            (self.head_dim + self.v_head_dim)
            * self.head_num
            * self.layer_num
            * self.dtype.itemsize
        )

    def get_ksize_per_token(self):
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize

    def init_kv_buffer(self):
        if self.layout == "page_first":
            k_dims = (self.size, self.layer_num, self.head_num, self.head_dim)
            v_dims = (self.size, self.layer_num, self.head_num, self.v_head_dim)
        elif self.layout == "page_first_direct":
            k_dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
            v_dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.v_head_dim,
            )
        else:
            raise ValueError(
                f"Unsupported layout for models with head_dim != v_head_dim: "
                f"{self.layout}; expected 'page_first' or 'page_first_direct'."
            )

        # token_stride_size / layout_dim are intentionally NOT set: K and V
        # have different strides, so any caller that reaches for a single
        # shared stride is a bug. Such callers will fail loudly with
        # AttributeError rather than silently use the K stride for V copies.

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        k_buffer = alloc_func(
            k_dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=self.page_size * self._k_layout_dim(),
        )
        v_buffer = alloc_func(
            v_dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=self.page_size * self._v_layout_dim(),
        )
        return (k_buffer, v_buffer)

    def _k_token_stride_size(self) -> int:
        return self.head_num * self.head_dim * self.dtype.itemsize

    def _v_token_stride_size(self) -> int:
        return self.head_num * self.v_head_dim * self.dtype.itemsize

    def _k_layout_dim(self) -> int:
        return self._k_token_stride_size() * self.layer_num

    def _v_layout_dim(self) -> int:
        return self._v_token_stride_size() * self.layer_num

    def _flat_page_unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "Models with head_dim != v_head_dim do not support the flat-page "
            "interface used by HiCache L3 storage backends {hf3fs, eic, nixl}. "
            "Use a backend that does not use this interface (e.g. mooncake, simm)."
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if self.device_pool is not None:
            if not is_draft and not self._is_device_layer_owned(device_pool, layer_id):
                return
            # MTP draft layers do not participate in CP layer sharding.
            host_layer_id = layer_id if is_draft else self._host_layer_index(layer_id)
            device_layer_id = 0 if is_draft else layer_id
        else:
            host_layer_id = device_layer_id = layer_id

        if io_backend == "kernel":
            if self.layout != "page_first":
                raise ValueError(
                    f"Unsupported layout for models with head_dim != v_head_dim "
                    f"and io_backend='kernel': {self.layout}; expected 'page_first'."
                )
            transfer_kv_per_layer_mla_pf_lf(
                src=self.k_buffer,
                dst=device_pool.k_buffer[device_layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=host_layer_id,
                item_size=self._k_token_stride_size(),
                src_layout_dim=self._k_layout_dim(),
            )
            transfer_kv_per_layer_mla_pf_lf(
                src=self.v_buffer,
                dst=device_pool.v_buffer[device_layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=host_layer_id,
                item_size=self._v_token_stride_size(),
                src_layout_dim=self._v_layout_dim(),
            )
        elif io_backend == "direct":
            if self.layout != "page_first_direct":
                raise ValueError(
                    f"Unsupported layout for models with head_dim != v_head_dim "
                    f"and io_backend='direct': {self.layout}; expected "
                    "'page_first_direct'."
                )
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.k_buffer],
                dst_ptrs=[device_pool.k_buffer[device_layer_id]],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=host_layer_id,
                page_size=self.page_size,
            )
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.v_buffer],
                dst_ptrs=[device_pool.v_buffer[device_layer_id]],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=host_layer_id,
                page_size=self.page_size,
            )
        else:
            raise ValueError(
                f"Unsupported IO backend for models with head_dim != v_head_dim: "
                f"{io_backend}; expected 'kernel' or 'direct'."
            )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        (
            device_k_data_ptrs,
            device_v_data_ptrs,
            device_k_buffers,
            device_v_buffers,
        ) = self._resolve_device_transfer_buffers(device_pool)
        if io_backend == "kernel":
            if self.layout != "page_first":
                raise ValueError(
                    f"Unsupported layout for models with head_dim != v_head_dim "
                    f"and io_backend='kernel': {self.layout}; expected 'page_first'."
                )
            if self.can_use_write_back_jit:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=device_k_data_ptrs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    staging=self.staging_k_buffer,
                    dst=self.k_buffer,
                    page_size=self.page_size,
                )
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=device_v_data_ptrs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    staging=self.staging_v_buffer,
                    dst=self.v_buffer,
                    page_size=self.page_size,
                )
            else:
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=device_k_data_ptrs,
                    dst=self.k_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self._k_token_stride_size(),
                    dst_layout_dim=self._k_layout_dim(),
                    num_layers=self.layer_num,
                )
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=device_v_data_ptrs,
                    dst=self.v_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self._v_token_stride_size(),
                    dst_layout_dim=self._v_layout_dim(),
                    num_layers=self.layer_num,
                )
        elif io_backend == "direct":
            if self.layout != "page_first_direct":
                raise ValueError(
                    f"Unsupported layout for models with head_dim != v_head_dim "
                    f"and io_backend='direct': {self.layout}; expected "
                    "'page_first_direct'."
                )
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=device_k_buffers,
                dst_ptrs=[self.k_buffer],
                src_indices=device_indices,
                dst_indices=host_indices,
                page_size=self.page_size,
            )
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=device_v_buffers,
                dst_ptrs=[self.v_buffer],
                src_indices=device_indices,
                dst_indices=host_indices,
                page_size=self.page_size,
            )
        else:
            raise ValueError(
                f"Unsupported IO backend for models with head_dim != v_head_dim: "
                f"{io_backend}; expected 'kernel' or 'direct'."
            )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise self._flat_page_unsupported()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise self._flat_page_unsupported()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise self._flat_page_unsupported()

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        raise NotImplementedError(
            "get_split_heads_page_buffer_meta requires layout='page_head', "
            "which is not supported for models with head_dim != v_head_dim."
        )

    def get_page_buffer_meta(self, indices):
        assert len(indices) % self.page_size == 0
        if self.layout not in ("page_first", "page_first_direct"):
            raise ValueError(
                f"Unsupported layout for models with head_dim != v_head_dim: "
                f"{self.layout}"
            )
        indices = indices.tolist()
        k_base_ptr = self.k_buffer.data_ptr()
        v_base_ptr = self.v_buffer.data_ptr()
        k_element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        v_element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.v_head_dim
        )
        ptr_list = []
        element_size_list = []
        if self.layout == "page_first_direct":
            k_index_stride = (
                self.layer_num * self.page_size * self.head_num * self.head_dim
            )
            v_index_stride = (
                self.layer_num * self.page_size * self.head_num * self.v_head_dim
            )
        else:
            k_index_stride = self.layer_num * self.head_num * self.head_dim
            v_index_stride = self.layer_num * self.head_num * self.v_head_dim
        for index in range(0, len(indices), self.page_size):
            buffer_index = (
                indices[index] // self.page_size
                if self.layout == "page_first_direct"
                else indices[index]
            )
            k_ptr = k_base_ptr + buffer_index * k_index_stride * self.dtype.itemsize
            v_ptr = v_base_ptr + buffer_index * v_index_stride * self.dtype.itemsize
            ptr_list.extend([k_ptr, v_ptr])
            element_size_list.extend([k_element_size, v_element_size])
        return ptr_list, element_size_list

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ("page_first", "page_first_direct"):
            return False
        k_stride = (
            self.page_size
            * self.layer_num
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        v_stride = (
            self.page_size
            * self.layer_num
            * self.head_num
            * self.v_head_dim
            * self.dtype.itemsize
        )
        base_aligned = (
            self.k_buffer.data_ptr() % page_size_bytes == 0
            and self.v_buffer.data_ptr() % page_size_bytes == 0
        )
        return (
            base_aligned
            and k_stride % page_size_bytes == 0
            and v_stride % page_size_bytes == 0
        )


def get_mha_host_pool_cls(device_pool: MHATokenToKVPool) -> type:
    """Pick the right MHA host-pool class based on the device pool's K/V dims.

    Returns ``AsymmetricMHATokenToKVPoolHost`` when ``head_dim != v_head_dim``
    (e.g. MiMo-V2), else the default ``MHATokenToKVPoolHost``.
    """
    # 2026-09-20 自前追加: INT2 は 1トークンが packed codes と scales/zeros の
    # 2テンソルに分かれ、packed は head_dim ではなく head_dim//4 バイトしかない。
    # 素の MHATokenToKVPoolHost は 4倍のバイト数を読んで範囲外アクセスで落ちる。
    if getattr(device_pool, "dtype", None) == "int2":
        return Int2MHATokenToKVPoolHost
    if device_pool.head_dim != device_pool.v_head_dim:
        return AsymmetricMHATokenToKVPoolHost
    return MHATokenToKVPoolHost
