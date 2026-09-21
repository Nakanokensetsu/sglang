import json
from abc import ABC, abstractmethod
from array import array
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import dill
import orjson
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """
    data = orjson.loads(json_str)
    return dill.loads(bytes.fromhex(data["callable"]))


class CustomLogitProcessor(ABC):
    """Abstract base class for callable functions."""

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Define the callable behavior."""
        raise NotImplementedError

    @classmethod
    def to_str(cls) -> str:
        """Serialize the callable function to a JSON-compatible string."""
        return json.dumps({"callable": dill.dumps(cls).hex()})

    @classmethod
    def from_str(cls, json_str: str):
        """Deserialize a callable function from a JSON string."""
        return _cache_from_str(json_str)()


class DisallowedTokensLogitsProcessor(CustomLogitProcessor):
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        disallowed_token_ids = custom_param_list[0]["token_ids"]
        assert all(
            disallowed_token_ids == c["token_ids"] for c in custom_param_list
        ), f"{custom_param_list=}"
        logits[..., disallowed_token_ids] = -float("inf")
        return logits


def _open_thinking_start(ids: list[int], start_id: int, end_id: int) -> int:
    """Return the index of the start token of the currently open thinking block, or -1."""
    for idx in reversed(range(len(ids))):
        if ids[idx] == start_id:
            return idx
        if ids[idx] == end_id:
            return -1
    return -1


class ThinkingBudgetLogitProcessor(CustomLogitProcessor):
    """A logit processor that controls the length of thinking."""

    THINKING_START_TOKEN_ID: int
    THINKING_END_TOKEN_ID: int
    NEW_LINE_TOKEN_ID: int

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):
        if custom_param_list is None or not custom_param_list:
            return logits
        for i, param_dict in enumerate(custom_param_list):
            if param_dict is None:
                continue

            thinking_budget: int | None = param_dict.get("thinking_budget")

            # Skip if thinking_budget is unset, or not an integer, or negative
            if (
                thinking_budget is None
                or not isinstance(thinking_budget, int)
                or thinking_budget < 0
            ):
                continue
            req: Req = param_dict.get("__req__")
            cur_ids: list[int] = [*req.origin_input_ids, *req.output_ids]

            # Check if out of thinking stage
            start_index = _open_thinking_start(
                cur_ids, self.THINKING_START_TOKEN_ID, self.THINKING_END_TOKEN_ID
            )
            if start_index < 0:
                continue

            # Count the number of tokens after the thinking start token
            num_tokens_after_start = len(cur_ids) - start_index - 1

            if num_tokens_after_start < thinking_budget:
                continue

            # Ensure new line token before thinking end token
            if not req.output_ids or req.output_ids[-1] != self.NEW_LINE_TOKEN_ID:
                logits[i, :] = -float("inf")
                logits[i, self.NEW_LINE_TOKEN_ID] = 0.0
                continue

            # Assign highest probability to the thinking end token
            logits[i, :] = -float("inf")
            logits[i, self.THINKING_END_TOKEN_ID] = 0.0

        return logits


class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for GLM-4.5 / GLM-4.6 / GLM-4.5V / GLM-4.6V models."""

    THINKING_START_TOKEN_ID: int = 151350
    THINKING_END_TOKEN_ID: int = 151351
    NEW_LINE_TOKEN_ID: int = 198


class Qwen3ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for Qwen3 models."""

    THINKING_START_TOKEN_ID: int = 151667
    THINKING_END_TOKEN_ID: int = 151668
    NEW_LINE_TOKEN_ID: int = 198


class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for DeepSeek-R1 models."""

    THINKING_START_TOKEN_ID: int = 128798
    THINKING_END_TOKEN_ID: int = 128799
    NEW_LINE_TOKEN_ID: int = 201


class InklingThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for Inkling models."""

    THINKING_START_TOKEN_ID: int = 200008
    THINKING_END_TOKEN_ID: int = 200010
    NEW_LINE_TOKEN_ID: int = 198


# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py
class DeepseekOCRNoRepeatNGramLogitProcessor(CustomLogitProcessor):
    """Block n-gram repetitions within a sliding window for DeepSeek-OCR outputs."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req = params.get("__req__")
            if req is None:
                continue

            try:
                ngram_size = int(params.get("ngram_size") or 0)
                window_size = int(params.get("window_size") or 0)
            except (TypeError, ValueError):
                continue

            if ngram_size <= 0 or window_size <= 0:
                continue

            sequence = req.origin_input_ids + req.output_ids
            if len(sequence) < ngram_size:
                continue

            search_start = max(0, len(sequence) - window_size)
            search_end = len(sequence) - ngram_size + 1
            if search_end <= search_start:
                continue

            if ngram_size > 1:
                current_prefix = sequence[-(ngram_size - 1) :]
            else:
                current_prefix = array("q")

            banned_tokens: Set[int] = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx : idx + ngram_size]
                if ngram_size == 1 or ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])

            whitelist_ids = params.get("whitelist_token_ids") or []
            try:
                whitelist = {int(token_id) for token_id in whitelist_ids}
            except (TypeError, ValueError):
                whitelist = set()

            banned_tokens.difference_update(whitelist)

            if not banned_tokens:
                continue

            indices = list(banned_tokens)
            logits[batch_idx, indices] = -float("inf")

        return logits


class Qwen38ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """Thinking-budget control for the Qwen3.5/3.8 family (248k vocab).

    The Qwen3 subclass above carries the OLD 151k-vocab ids; the 3.5/3.8
    family retokenized (<think>=248068, </think>=248069) and silently never
    triggers under it. Newline stays 198.
    """

    THINKING_START_TOKEN_ID: int = 248068
    THINKING_END_TOKEN_ID: int = 248069
    NEW_LINE_TOKEN_ID: int = 198


# --- 思考予算プロセッサの解決(2026-09-19 追加) -------------------------------
# sglang にはモデル別サブクラスが並んでいるだけで、「どれを使うか」を選ぶ配線が
# 無い(Glm4Moe/Qwen3/DeepSeekR1/Inkling のいずれも定義ファイル以外からの参照が0件)。
# そのため呼び出し側が custom_logit_processor に callable を明示しない限り
# custom_params.thinking_budget は黙って無視される(上流 issue #25536 と同じ穴)。
# tokenizer から <think>/</think> の実 ID を引いて対応クラスを選ぶ。
# 一致が無ければその場でサブクラスを生成するので、新モデルでも ID の
# ハードコードを足し直す必要がない。
_THINK_TOKEN_CANDIDATES = (("<think>", "</think>"), ("<thinking>", "</thinking>"))
_NEWLINE_FALLBACK_ID = 198


def _think_token_ids(tokenizer):
    if tokenizer is None:
        return None
    for start_tok, end_tok in _THINK_TOKEN_CANDIDATES:
        try:
            sid = tokenizer.convert_tokens_to_ids(start_tok)
            eid = tokenizer.convert_tokens_to_ids(end_tok)
        except Exception:
            continue
        unk = getattr(tokenizer, "unk_token_id", None)
        if sid is None or eid is None or sid == unk or eid == unk or sid == eid:
            continue
        try:
            nid = tokenizer.convert_tokens_to_ids("\n")
            if nid is None or nid == unk:
                nid = _NEWLINE_FALLBACK_ID
        except Exception:
            nid = _NEWLINE_FALLBACK_ID
        return int(sid), int(eid), int(nid)
    return None


@lru_cache(maxsize=8)
def _thinking_budget_cls_for(start_id: int, end_id: int, newline_id: int):
    for cls in ThinkingBudgetLogitProcessor.__subclasses__():
        if (
            getattr(cls, "THINKING_START_TOKEN_ID", None) == start_id
            and getattr(cls, "THINKING_END_TOKEN_ID", None) == end_id
            and getattr(cls, "NEW_LINE_TOKEN_ID", None) == newline_id
        ):
            return cls
    return type(
        "ResolvedThinkingBudgetLogitProcessor",
        (ThinkingBudgetLogitProcessor,),
        {
            "THINKING_START_TOKEN_ID": start_id,
            "THINKING_END_TOKEN_ID": end_id,
            "NEW_LINE_TOKEN_ID": newline_id,
        },
    )


def resolve_thinking_budget_processor(tokenizer):
    """このモデルに合う思考予算プロセッサを返す。解決できなければ None。"""
    ids = _think_token_ids(tokenizer)
    if ids is None:
        return None
    return _thinking_budget_cls_for(*ids)
