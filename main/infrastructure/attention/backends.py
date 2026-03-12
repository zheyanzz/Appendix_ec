
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AttentionBackend:
    name: str
    fn: Callable[[Tensor, Tensor, Tensor], Tensor]
    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return self.fn(q, k, v)

def _supports_flash(q: Tensor) -> bool:
    return q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)


def _flash_attention_fn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    from flash_attn import flash_attn_func

    if q.dim() != 4:
        raise ValueError("flash attention expects rank-4 tensors")
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    out = flash_attn_func(q_t, k_t, v_t, causal=False)
    return out.transpose(1, 2).contiguous()


def _xformers_attention_fn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:

    from xformers.ops import memory_efficient_attention

    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    out = memory_efficient_attention(q_t, k_t, v_t)
    return out.transpose(1, 2).contiguous()


def _sdpa_attention_fn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return F.scaled_dot_product_attention(q, k, v)


def _manual_attention_fn(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    d = q.shape[-1]
    attn = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def _resolve_backend_name(requested: str) -> str:

    name = (requested or "auto").lower()
    if name not in {"auto", "flash", "xformers", "sdpa", "manual"}:
        raise ValueError(f"Unknown attention backend: {requested}")
    return name


def get_attention_backend(requested: str = "auto") -> AttentionBackend:
    name = _resolve_backend_name(requested)


    if name == "manual":
        return AttentionBackend(name="manual", fn=_manual_attention_fn)
    if name == "sdpa":
        return AttentionBackend(name="sdpa", fn=_sdpa_attention_fn)
    if name == "flash":
        return AttentionBackend(name="flash", fn=_flash_attention_fn)
    if name == "xformers":
        return AttentionBackend(name="xformers", fn=_xformers_attention_fn)


    

    def auto_backend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:

        if _supports_flash(q):
            try:
                return _flash_attention_fn(q, k, v)
            except Exception:
                pass

        try:
            return _xformers_attention_fn(q, k, v)
        except Exception:
            pass

        if hasattr(F, "scaled_dot_product_attention"):
            return _sdpa_attention_fn(q, k, v)
        return _manual_attention_fn(q, k, v)
    return AttentionBackend(name="auto", fn=auto_backend)

