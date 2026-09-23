from __future__ import annotations

import json
import logging
import os
import platform
from collections import defaultdict

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import alloc_mmap
from sglang.srt.runtime_context import get_memory

logger = logging.getLogger(__name__)

_CUDA_HOST_REGISTERED_RANGES_ATTR = "_sglang_cuda_host_registered_ranges"


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class ShmHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        from sglang.srt.mem_cache.storage.mmap import alloc_shm

        tensor, fd, mm = alloc_shm(dims, dtype)
        self.fds.append(fd)
        self.mms.append(mm)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type() -> str:
    """The host-allocator kind the published HiCache configuration asks for."""

    backend = get_memory().hicache_storage_backend
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = get_memory().hicache_storage_backend_extra_config
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_host_register(
    buffer: torch.Tensor, registration_granularity_bytes: int | None = None
) -> None:
    # Avoid oversized cudaHostRegister calls on large host pools.
    cudart = torch.cuda.cudart()
    base = buffer.data_ptr()
    total = buffer.numel() * buffer.element_size()
    chunk_limit_bytes = (
        max(envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB.get(), 1) * 1024**3
    )
    # Preserve the legacy single-call behavior unless the caller provides a
    # copy granularity. Splitting an unknown page-first layout at an arbitrary
    # byte offset can make one cudaMemcpyBatchAsync span two registrations.
    chunk_bytes = total
    if registration_granularity_bytes is not None:
        if registration_granularity_bytes <= 0:
            raise ValueError(
                "registration_granularity_bytes must be positive, got "
                f"{registration_granularity_bytes}"
            )
        if registration_granularity_bytes > chunk_limit_bytes:
            raise ValueError(
                "Host registration granularity exceeds the configured chunk limit: "
                f"granularity={registration_granularity_bytes}, "
                f"chunk_limit={chunk_limit_bytes}"
            )
        chunk_bytes = (
            chunk_limit_bytes // registration_granularity_bytes
        ) * registration_granularity_bytes
    registered_ranges: list[tuple[int, int]] = []
    try:
        offset = 0
        while offset < total:
            size = min(chunk_bytes, total - offset)
            ptr = base + offset
            rc = int(cudart.cudaHostRegister(ptr, size, 0))
            if rc != 0:
                # 2026-09-20 自前修正: cudaGetErrorString は
                # torch._C._cudart.cudaError しか受け取らない。int を渡すと
                # TypeError になり、**本来の失敗理由が握り潰される**。
                try:
                    rc_msg = cudart.cudaGetErrorString(cudart.cudaError(rc))
                except Exception:
                    rc_msg = "unknown"
                raise RuntimeError(
                    f"cudaHostRegister failed (rc={rc}, "
                    f"{rc_msg}) at offset={offset} size={size} "
                    f"(total={total}, chunk_limit={chunk_bytes}); host buffer is not "
                    f"pinned and device transfers may silently return stale data."
                )
            registered_ranges.append((ptr, size))
            offset += size

        # Keep the exact registration bases alive with the tensor. CUDA requires
        # cudaHostUnregister to receive each base pointer, not just the tensor's
        # original base once after several independent registrations.
        setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, registered_ranges)
    except Exception:
        remaining_ranges = _cuda_host_unregister_ranges(
            cudart, registered_ranges, operation="registration rollback"
        )
        if remaining_ranges:
            setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)
        raise


def _cuda_host_unregister_ranges(
    cudart, registered_ranges: list[tuple[int, int]], *, operation: str
) -> list[tuple[int, int]]:
    failed_ranges = []
    for ptr, size in reversed(registered_ranges):
        rc = int(cudart.cudaHostUnregister(ptr))
        if rc != 0:
            failed_ranges.append((ptr, size))
            logger.warning(
                "cudaHostUnregister failed during %s (rc=%d, %s) "
                "for ptr=%#x size=%d",
                operation,
                rc,
                cudart.cudaGetErrorString(rc),
                ptr,
                size,
            )
    failed_ranges.reverse()
    return failed_ranges


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    registered_ranges = getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, None)
    if registered_ranges is None:
        # Compatibility for buffers registered before range metadata was added.
        registered_ranges = [
            (buffer.data_ptr(), buffer.numel() * buffer.element_size())
        ]
    if not registered_ranges:
        return

    remaining_ranges = _cuda_host_unregister_ranges(
        cudart, registered_ranges, operation="host-pool destroy"
    )
    setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer, registration_granularity_bytes)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


def _host_register_is_mappable() -> bool:
    """``cudaHostRegister`` した領域をカーネルが直接触れるか。

    2026-09-20 自前追加。HiCache の mamba バックアップカーネル
    (``transfer_mamba.cuh`` の ``transfer_mamba_backup_kernel``)は、
    ホストプールの**ホストポインタへ GPU から直接ストア**する。これは登録領域が
    デバイスのアドレス空間にマップされていることが前提になる。

    **WSL2 ではこの前提が成り立たない。** malloc したホストメモリは
    ``cudaHostRegister`` しても(flags を Default/Portable/Mapped/両方の
    どれにしても)マップされず、カーネルが触った瞬間に
    ``illegal memory access`` になる。``cudaHostAlloc``(= torch の
    ``pin_memory=True``)で確保したものは問題なく触れる。実測で切り分け済み:

        dst=GPU              -> OK
        dst=torch pin_memory -> OK
        dst=cudaHostRegister -> illegal memory access (flags 0/1/2/3 全滅)
        dst=未ピン            -> illegal memory access

    再現手順は ``tools/mamba_transfer_repro.py``。

    ``SGLANG_HICACHE_HOST_ALLOC`` で明示上書きできる(``pin`` / ``register``)。
    """
    override = os.environ.get("SGLANG_HICACHE_HOST_ALLOC", "").strip().lower()
    if override == "pin":
        return False
    if override == "register":
        return True
    # WSL 判定。マイクロソフトのカーネルは release に "microsoft" を含む。
    try:
        return "microsoft" not in platform.uname().release.lower()
    except Exception:
        return True


def _alloc_cuda_host(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    if _host_register_is_mappable():
        return alloc_with_host_register(
            dims,
            dtype,
            device,
            pin_memory,
            allocator,
            registration_granularity_bytes,
        )
    # cudaHostAlloc 経路。``allocator``(mmap/shm)は使えないので、
    # 別アロケータを要求されている場合は黙って無視せず落とす。
    allocator_name = type(allocator).__name__ if allocator is not None else "None"
    # 既定の ``HostTensorAllocator`` は torch.empty を呼ぶだけなので
    # cudaHostAlloc 経路で置き換えられる。mmap/shm/mooncake 等は置き換えられない。
    if allocator is not None and allocator_name != "HostTensorAllocator":
        raise RuntimeError(
            "HiCache on this platform must allocate host pools with "
            f"cudaHostAlloc, which cannot use the {allocator_name} allocator "
            "(--hicache-storage-backend shm/... is unsupported here). "
            "Set SGLANG_HICACHE_HOST_ALLOC=register to override at your own risk."
        )
    if not getattr(_alloc_cuda_host, "_logged", False):
        _alloc_cuda_host._logged = True
        logger.info(
            "HiCache host pools use cudaHostAlloc (pin_memory) instead of "
            "cudaHostRegister: registered host memory is not device-mappable "
            "on this platform (WSL2). Override with SGLANG_HICACHE_HOST_ALLOC."
        )
    return alloc_with_pin_memory(
        dims, dtype, device, pin_memory, None, registration_granularity_bytes
    )


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: _alloc_cuda_host,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
