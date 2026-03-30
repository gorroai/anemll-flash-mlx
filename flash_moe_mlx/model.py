from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from .expert_io import (
    ExpertGeometry,
    ExpertLoaderProtocol,
    ExpertSlotView,
    MixedPrecisionSidecarLoader,
    NativeExpertLoader,
    NativeSlotBankLoader,
    ResidentFlashExpertLoader,
    unpack_expert_slot,
)
from .upstream_gated_delta import gated_delta_update as official_gated_delta_update
from .upstream_switch_layers import QuantizedSwitchGLUExecutor, SwitchGLU


@dataclass
class ModelArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    shared_expert_intermediate_size: int
    moe_intermediate_size: int
    intermediate_size: int
    mlp_only_layers: List[int]
    layer_types: List[str]
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    full_attention_interval: int
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None
    quant_group_size: int = 64
    quant_bits: int = 4
    quant_mode: str = "affine"
    quantization_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    eos_token_ids: tuple[int, ...] = ()

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "ModelArgs":
        with (model_dir / "config.json").open("r", encoding="utf-8") as f:
            raw = json.load(f)

        text_cfg = raw["text_config"]
        rope_params = dict(text_cfg.get("rope_parameters") or {})
        quant = raw.get("quantization") or raw.get("quantization_config") or {}
        quant_overrides: dict[str, dict[str, Any]] = {}
        for key, value in quant.items():
            if key in {"bits", "group_size"}:
                continue
            if isinstance(value, dict):
                quant_overrides[key] = {
                    "group_size": int(value.get("group_size", quant.get("group_size", 64))),
                    "bits": int(value.get("bits", quant.get("bits", 4))),
                    "mode": str(value.get("mode", quant.get("mode", "affine"))),
                }

        eos = raw.get("eos_token_id", text_cfg.get("eos_token_id"))
        if isinstance(eos, int):
            eos_ids = (eos,)
        else:
            eos_ids = tuple(int(token_id) for token_id in eos)

        return cls(
            model_type=text_cfg.get("model_type", raw.get("model_type", "qwen3_5_moe_text")),
            hidden_size=int(text_cfg["hidden_size"]),
            num_hidden_layers=int(text_cfg["num_hidden_layers"]),
            num_attention_heads=int(text_cfg["num_attention_heads"]),
            num_key_value_heads=int(text_cfg["num_key_value_heads"]),
            head_dim=int(text_cfg["head_dim"]),
            linear_num_value_heads=int(text_cfg["linear_num_value_heads"]),
            linear_num_key_heads=int(text_cfg["linear_num_key_heads"]),
            linear_key_head_dim=int(text_cfg["linear_key_head_dim"]),
            linear_value_head_dim=int(text_cfg["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(text_cfg["linear_conv_kernel_dim"]),
            num_experts=int(text_cfg["num_experts"]),
            num_experts_per_tok=int(text_cfg["num_experts_per_tok"]),
            decoder_sparse_step=int(text_cfg.get("decoder_sparse_step", 1)),
            shared_expert_intermediate_size=int(text_cfg["shared_expert_intermediate_size"]),
            moe_intermediate_size=int(text_cfg["moe_intermediate_size"]),
            intermediate_size=int(
                text_cfg.get("intermediate_size", text_cfg["shared_expert_intermediate_size"])
            ),
            mlp_only_layers=list(text_cfg.get("mlp_only_layers", [])),
            layer_types=list(text_cfg.get("layer_types", [])),
            rms_norm_eps=float(text_cfg["rms_norm_eps"]),
            vocab_size=int(text_cfg["vocab_size"]),
            rope_theta=float(rope_params.get("rope_theta", text_cfg.get("rope_theta", 1000000.0))),
            partial_rotary_factor=float(
                rope_params.get("partial_rotary_factor", text_cfg.get("partial_rotary_factor", 1.0))
            ),
            max_position_embeddings=int(text_cfg["max_position_embeddings"]),
            full_attention_interval=int(text_cfg.get("full_attention_interval", 4)),
            norm_topk_prob=bool(text_cfg.get("norm_topk_prob", True)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            attention_bias=bool(text_cfg.get("attention_bias", False)),
            rope_scaling=rope_params or None,
            quant_group_size=int(quant.get("group_size", 64)),
            quant_bits=int(quant.get("bits", 4)),
            quant_mode=str(quant.get("mode", "affine")),
            quantization_overrides=quant_overrides,
            eos_token_ids=eos_ids,
        )


@dataclass
class ModelBundle:
    model: "TextOnlyQwen35"
    tokenizer: Tokenizer
    config: ModelArgs
    expert_loader: Optional[ExpertLoaderProtocol]
    expert_geometry: ExpertGeometry
    expert_bits: int = 4
    expert_mode: str = "affine"
    resident_pread_mlx: bool = False
    resident_rebind: bool = False
    resident_copy_k: bool = False
    slot_bank_size: int = 0
    slot_bank_native: bool = False
    bypass_routed_mlp: bool = False


@dataclass(frozen=True)
class GenerationStats:
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float

    @property
    def prefill_tokens_per_second(self) -> float:
        if self.prefill_seconds <= 0.0:
            return 0.0
        return self.prompt_tokens / self.prefill_seconds

    @property
    def decode_tokens_per_second(self) -> float:
        if self.decode_seconds <= 0.0:
            return 0.0
        return self.generated_tokens / self.decode_seconds


@dataclass
class RebindStageStats:
    calls: int = 0
    selector_seconds: float = 0.0
    take_seconds: float = 0.0
    mutate_seconds: float = 0.0
    forward_seconds: float = 0.0
    combine_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return (
            self.selector_seconds
            + self.take_seconds
            + self.mutate_seconds
            + self.forward_seconds
            + self.combine_seconds
        )

    def reset(self) -> None:
        self.calls = 0
        self.selector_seconds = 0.0
        self.take_seconds = 0.0
        self.mutate_seconds = 0.0
        self.forward_seconds = 0.0
        self.combine_seconds = 0.0


@dataclass
class SlotBankStageStats:
    calls: int = 0
    hit_calls: int = 0
    miss_calls: int = 0
    selector_seconds: float = 0.0
    lookup_seconds: float = 0.0
    install_seconds: float = 0.0
    index_seconds: float = 0.0
    forward_seconds: float = 0.0
    combine_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return (
            self.selector_seconds
            + self.lookup_seconds
            + self.install_seconds
            + self.index_seconds
            + self.forward_seconds
            + self.combine_seconds
        )

    def reset(self) -> None:
        self.calls = 0
        self.hit_calls = 0
        self.miss_calls = 0
        self.selector_seconds = 0.0
        self.lookup_seconds = 0.0
        self.install_seconds = 0.0
        self.index_seconds = 0.0
        self.forward_seconds = 0.0
        self.combine_seconds = 0.0


def create_causal_mask(
    n_tokens: int,
    offset: int = 0,
    window_size: Optional[int] = None,
):
    right_indices = mx.arange(offset + n_tokens)
    left_indices = mx.arange(offset, offset + n_tokens) if offset else right_indices
    left_indices = left_indices[:, None]
    right_indices = right_indices[None]
    mask = left_indices >= right_indices
    if window_size is not None:
        mask = mask & (left_indices < right_indices + window_size)
    return mask


def create_attention_mask(
    hidden_states: mx.array,
    cache: Optional["KVCache"] = None,
    window_size: Optional[int] = None,
    return_array: bool = False,
):
    n_tokens = hidden_states.shape[1]
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(n_tokens, return_array=return_array, window_size=window_size)
    if n_tokens == 1:
        return None
    if return_array or (window_size and n_tokens > window_size):
        return create_causal_mask(n_tokens, window_size=window_size)
    return "causal"


def create_ssm_mask(hidden_states: mx.array, cache: Optional["ArraysCache"] = None):
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(hidden_states.shape[1])
    return None


def scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: Optional["KVCache"],
    scale: float,
    mask: Optional[mx.array],
):
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
    )


class KVCache:
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            batch, num_heads, _, key_dim = keys.shape
            value_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            key_shape = (batch, num_heads, n_steps * self.step, key_dim)
            value_shape = (batch, num_heads, n_steps * self.step, value_dim)
            new_keys = mx.zeros(key_shape, keys.dtype)
            new_values = mx.zeros(value_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_keys], axis=2)
                self.values = mx.concatenate([self.values, new_values], axis=2)
            else:
                self.keys = new_keys
                self.values = new_values

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def make_mask(self, n_tokens: int, return_array: bool = False, window_size: Optional[int] = None):
        return create_attention_mask(
            mx.zeros((1, n_tokens, 1)),
            cache=None,
            return_array=return_array,
            window_size=window_size,
        ) if self.offset == 0 else create_causal_mask(n_tokens, offset=self.offset, window_size=window_size)


class ArraysCache:
    def __init__(self, size: int):
        self.cache = [None] * size

    def __getitem__(self, index: int):
        return self.cache[index]

    def __setitem__(self, index: int, value):
        self.cache[index] = value

    def make_mask(self, n_tokens: int):
        return None


class MambaCache(ArraysCache):
    def __init__(self):
        super().__init__(size=2)


class Qwen35RotaryEmbedding:
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        mrope_section: Sequence[int] = (11, 11, 0),
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim))
        self.inv_freq = inv_freq
        self.mrope_section = list(mrope_section)

    def apply_interleaved_mrope(self, freqs: mx.array, mrope_section: Sequence[int]) -> mx.array:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            index = slice(offset, length, 3)
            freqs_t[..., index] = freqs[dim, ..., index]
        return freqs_t

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )
        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)
        freqs = inv_freq_expanded @ position_ids_expanded
        freqs = mx.swapaxes(freqs, 2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multimodal_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    unsqueeze_dim: int = 1,
) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return mx.concatenate([q_embed, q_pass], axis=-1), mx.concatenate([k_embed, k_pass], axis=-1)


def _sample_next_token(logits: mx.array, temperature: float, rng: np.random.Generator) -> int:
    logits_np = np.array(logits.astype(mx.float32)).reshape(-1)
    if temperature <= 0.0:
        return int(np.argmax(logits_np))
    scaled = logits_np / max(temperature, 1e-6)
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def _apply_sparse_moe_tail(
    routed_y: mx.array,
    shared_gate_logits: mx.array,
    shared_y: mx.array,
) -> mx.array:
    return routed_y + (mx.sigmoid(shared_gate_logits) * shared_y)


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            x = nn.silu(gate) * x
        return x


class Qwen3TextAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        q_out_dim = self.num_attention_heads * self.head_dim * 2
        kv_out_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(args.hidden_size, q_out_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, kv_out_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, kv_out_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rotary_emb = Qwen35RotaryEmbedding(
            int(self.head_dim * args.partial_rotary_factor),
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            mrope_section=args.rope_scaling.get("mrope_section", [11, 11, 10]) if args.rope_scaling else [11, 11, 10],
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(batch, seq_len, self.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch, seq_len, -1)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(batch, seq_len, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(batch, seq_len, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        kv_seq_len = keys.shape[-2]
        if position_ids is None:
            if cache is None:
                base_positions = mx.arange(seq_len)
            else:
                kv_seq_len += cache.offset + 1
                base_positions = mx.arange(cache.offset, cache.offset + seq_len)
            position_ids = mx.expand_dims(base_positions, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        else:
            kv_seq_len += cache.offset + 1 if cache is not None else 0

        cos, sin = self.rotary_emb(values, position_ids)
        if mask is not None and isinstance(mask, mx.array):
            if isinstance(kv_seq_len, mx.array):
                kv_seq_len = kv_seq_len.max().item()
            mask = mask[..., : int(kv_seq_len)]
        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class Qwen3TextMLP(nn.Module):
    def __init__(self, args: ModelArgs, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, hidden_dim, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3TextGatedDeltaNet(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones(self.num_v_heads)
        self.A_log = mx.log(mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,)))
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[MambaCache] = None,
    ) -> mx.array:
        batch, seq_len, _ = inputs.shape
        mixed_qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(batch, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != batch:
                conv_state = mx.zeros(
                    (batch, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            if mask.shape[0] != batch:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            tensor.reshape(batch, seq_len, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        recurrent_state = cache[1] if cache is not None else None
        if recurrent_state is not None and recurrent_state.shape[0] != batch:
            recurrent_state = None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, recurrent_state = official_gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            recurrent_state,
            mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = recurrent_state
        out = self.norm(out, z)
        return self.out_proj(out.reshape(batch, seq_len, -1))


class Qwen3TextSparseMoeBlock(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_index: int,
        expert_loader: Optional[ExpertLoaderProtocol],
        expert_geometry: ExpertGeometry,
        routed_top_k: int,
        use_resident_experts: bool,
        use_resident_pread_mlx: bool,
        use_resident_rebind: bool,
        use_resident_copy_k: bool,
        slot_bank_size: int,
        expert_bits: int | tuple[int, int, int],
        expert_mode: str,
    ):
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.top_k = routed_top_k
        self.group_size = args.quant_group_size
        loader_bits = None
        if expert_loader is not None and hasattr(expert_loader, "layer_bits"):
            loader_bits = getattr(expert_loader, "layer_bits")(layer_index)
        if loader_bits is not None:
            if len(loader_bits) != 3:
                raise ValueError(f"layer_bits must have 3 entries, received {loader_bits!r}")
            self.gate_bits = int(loader_bits[0])
            self.up_bits = int(loader_bits[1])
            self.down_bits = int(loader_bits[2])
        elif isinstance(expert_bits, (tuple, list)):
            if len(expert_bits) != 3:
                raise ValueError(f"expert_bits tuple must have 3 entries, received {expert_bits!r}")
            self.gate_bits = int(expert_bits[0])
            self.up_bits = int(expert_bits[1])
            self.down_bits = int(expert_bits[2])
        else:
            uniform_bits = int(expert_bits)
            self.gate_bits = uniform_bits
            self.up_bits = uniform_bits
            self.down_bits = uniform_bits
        self.mode = expert_mode
        self.layer_index = layer_index
        self.expert_geometry = expert_geometry
        self._expert_loader = expert_loader
        self._use_resident_experts = use_resident_experts
        self._use_resident_pread_mlx = use_resident_pread_mlx
        self._use_resident_rebind = use_resident_rebind
        self._use_resident_copy_k = use_resident_copy_k
        self._slot_bank_size = slot_bank_size
        self._bypass_routed_mlp = False
        self._trace_callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None
        self._routing_sample_callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None
        self._candidate_trace_callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None
        self._candidate_trace_top_n = 0
        self._execution_mode_override: Optional[str] = None
        self._rebind_stage_timing_enabled = False
        self._rebind_stage_stats = RebindStageStats()
        self._slot_bank_stage_timing_enabled = False
        self._slot_bank_stage_stats = SlotBankStageStats()
        self._slot_bank_device_hit_lookup_enabled = False
        self._slot_bank_direct_contiguous_hit_enabled = False
        self._bank_index_sort_ids = False
        self._bank_index_use_compile = False
        self._compiled_tail_enabled = False
        self._compiled_tail_fn: Optional[Callable[[mx.array, mx.array, mx.array], mx.array]] = None

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = (
            SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.num_experts)
            if use_resident_experts
            else None
        )
        self._resident_packed_store = (
            QuantizedSwitchGLUExecutor(
                args.hidden_size,
                args.moe_intermediate_size,
                args.num_experts,
                group_size=self.group_size,
                gate_bits=self.gate_bits,
                up_bits=self.up_bits,
                down_bits=self.down_bits,
                mode=self.mode,
            )
            if use_resident_pread_mlx or use_resident_rebind or use_resident_copy_k
            else None
        )
        self._resident_rebind_executor = (
            QuantizedSwitchGLUExecutor(
                args.hidden_size,
                args.moe_intermediate_size,
                routed_top_k,
                group_size=self.group_size,
                gate_bits=self.gate_bits,
                up_bits=self.up_bits,
                down_bits=self.down_bits,
                mode=self.mode,
            )
            if use_resident_rebind
            else None
        )
        self._resident_copy_executor = (
            QuantizedSwitchGLUExecutor(
                args.hidden_size,
                args.moe_intermediate_size,
                routed_top_k,
                group_size=self.group_size,
                gate_bits=self.gate_bits,
                up_bits=self.up_bits,
                down_bits=self.down_bits,
                mode=self.mode,
            )
            if use_resident_copy_k
            else None
        )
        self._streamed_switch_executor = (
            None
            if use_resident_experts
            or use_resident_pread_mlx
            or use_resident_rebind
            or use_resident_copy_k
            or slot_bank_size > 0
            else QuantizedSwitchGLUExecutor(
                args.hidden_size,
                args.moe_intermediate_size,
                routed_top_k,
                group_size=self.group_size,
                gate_bits=self.gate_bits,
                up_bits=self.up_bits,
                down_bits=self.down_bits,
                mode=self.mode,
            )
        )
        self._slot_bank_executor = (
            QuantizedSwitchGLUExecutor(
                args.hidden_size,
                args.moe_intermediate_size,
                slot_bank_size,
                group_size=self.group_size,
                gate_bits=self.gate_bits,
                up_bits=self.up_bits,
                down_bits=self.down_bits,
                mode=self.mode,
            )
            if slot_bank_size > 0
            else None
        )
        self._slot_bank_lookup = (
            mx.full((args.num_experts,), -1, dtype=mx.int32) if slot_bank_size > 0 else None
        )
        self._slot_bank_owner: list[int] = [-1] * slot_bank_size
        self._slot_bank_expert_to_slot: dict[int, int] = {}
        self._slot_bank_lru: OrderedDict[int, None] = OrderedDict()
        self._slot_bank_requests = 0
        self._slot_bank_misses = 0
        self._slot_bank_full_hit_calls = 0
        self._slot_bank_calls = 0
        self._last_slot_bank_expert_ids: tuple[int, ...] = ()
        self.shared_expert = Qwen3TextMLP(args, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def load_packed_resident_experts(self, slot_buffers: Sequence[memoryview | ExpertSlotView]) -> None:
        if self._resident_packed_store is None:
            raise RuntimeError("Packed resident executor is not configured")
        if not slot_buffers:
            return
        first = slot_buffers[0]
        if isinstance(first, ExpertSlotView):
            experts = list(slot_buffers)
        else:
            experts = [unpack_expert_slot(slot_buffer, self.expert_geometry) for slot_buffer in slot_buffers]
        self._resident_packed_store.load_quantized_views(experts)

    def _load_expert_views(self, expert_ids: Sequence[int]) -> list[ExpertSlotView]:
        if self._expert_loader is None:
            raise RuntimeError("Expert loader is not configured")
        return self._expert_loader.load_layer_views(self.layer_index, list(expert_ids))

    def _resident_rebind_switch_mlp(
        self,
        x: mx.array,
        expert_indices: mx.array,
        expert_scores: mx.array,
    ) -> mx.array:
        if self._resident_packed_store is None or self._resident_rebind_executor is None:
            raise RuntimeError("Resident rebind path is not configured")
        if x.shape[0] != 1 or x.shape[1] != 1:
            raise NotImplementedError(
                "Resident rebind currently expects single-token execution so each token can "
                "rebind the selected expert tensors."
            )

        timing_enabled = self._rebind_stage_timing_enabled

        selector_start = time.perf_counter()
        token_experts = np.asarray(expert_indices.reshape(-1, self.top_k)[0], dtype=np.int32)
        selector = mx.array(token_experts, dtype=mx.int32)
        if timing_enabled:
            mx.eval(selector)
            self._rebind_stage_stats.selector_seconds += time.perf_counter() - selector_start

        take_start = time.perf_counter()
        gate_weight = mx.take(self._resident_packed_store.gate_proj.weight, selector, axis=0)
        gate_scales = mx.take(self._resident_packed_store.gate_proj.scales, selector, axis=0)
        gate_biases = mx.take(self._resident_packed_store.gate_proj.biases, selector, axis=0)
        up_weight = mx.take(self._resident_packed_store.up_proj.weight, selector, axis=0)
        up_scales = mx.take(self._resident_packed_store.up_proj.scales, selector, axis=0)
        up_biases = mx.take(self._resident_packed_store.up_proj.biases, selector, axis=0)
        down_weight = mx.take(self._resident_packed_store.down_proj.weight, selector, axis=0)
        down_scales = mx.take(self._resident_packed_store.down_proj.scales, selector, axis=0)
        down_biases = mx.take(self._resident_packed_store.down_proj.biases, selector, axis=0)
        if timing_enabled:
            mx.eval(
                gate_weight,
                gate_scales,
                gate_biases,
                up_weight,
                up_scales,
                up_biases,
                down_weight,
                down_scales,
                down_biases,
            )
            self._rebind_stage_stats.take_seconds += time.perf_counter() - take_start

        mutate_start = time.perf_counter()
        self._resident_rebind_executor.load_quantized(
            gate_weight=gate_weight,
            gate_scales=gate_scales,
            gate_biases=gate_biases,
            up_weight=up_weight,
            up_scales=up_scales,
            up_biases=up_biases,
            down_weight=down_weight,
            down_scales=down_scales,
            down_biases=down_biases,
        )
        if timing_enabled:
            self._rebind_stage_stats.mutate_seconds += time.perf_counter() - mutate_start

        forward_start = time.perf_counter()
        local_indices = mx.broadcast_to(mx.arange(self.top_k, dtype=mx.int32), expert_indices.shape)
        routed_y = self._resident_rebind_executor(x, local_indices)
        if timing_enabled:
            mx.eval(routed_y)
            self._rebind_stage_stats.forward_seconds += time.perf_counter() - forward_start

        combine_start = time.perf_counter()
        combined = (routed_y * expert_scores[..., None]).sum(axis=-2)
        if timing_enabled:
            mx.eval(combined)
            self._rebind_stage_stats.combine_seconds += time.perf_counter() - combine_start
            self._rebind_stage_stats.calls += 1
        return combined

    def _resident_copy_k_switch_mlp(
        self,
        x: mx.array,
        expert_indices: mx.array,
        expert_scores: mx.array,
    ) -> mx.array:
        if self._resident_packed_store is None or self._resident_copy_executor is None:
            raise RuntimeError("Resident copy-k path is not configured")
        if x.shape[0] != 1 or x.shape[1] != 1:
            raise NotImplementedError(
                "Resident copy-k currently expects single-token execution so each token can "
                "copy the selected expert tensors into stable K-slot buffers."
            )

        token_experts = np.asarray(expert_indices.reshape(-1, self.top_k)[0], dtype=np.int32).tolist()
        self._resident_copy_executor.copy_experts_from(self._resident_packed_store, token_experts)
        local_indices = mx.broadcast_to(mx.arange(self.top_k, dtype=mx.int32), expert_indices.shape)
        routed_y = self._resident_copy_executor(x, local_indices)
        return (routed_y * expert_scores[..., None]).sum(axis=-2)

    def _bank_index_switch_mlp(
        self,
        executor: QuantizedSwitchGLUExecutor,
        x: mx.array,
        expert_indices: mx.array,
        expert_scores: mx.array,
    ) -> mx.array:
        exec_indices = expert_indices
        exec_scores = expert_scores
        assume_sorted_indices = False
        if self._bank_index_sort_ids:
            sort_order = mx.argsort(expert_indices, axis=-1)
            exec_indices = mx.take_along_axis(expert_indices, sort_order, axis=-1)
            exec_scores = mx.take_along_axis(expert_scores, sort_order, axis=-1)
            assume_sorted_indices = True

        if self._bank_index_use_compile:
            routed_y = executor.compiled(x, exec_indices, assume_sorted_indices=assume_sorted_indices)
        else:
            routed_y = executor(x, exec_indices, assume_sorted_indices=assume_sorted_indices)
        return (routed_y * exec_scores[..., None]).sum(axis=-2)

    def _touch_slot_bank_slot(self, slot_id: int) -> None:
        if slot_id in self._slot_bank_lru:
            self._slot_bank_lru.move_to_end(slot_id)
        else:
            self._slot_bank_lru[slot_id] = None

    def reset_slot_bank_state(self) -> None:
        if self._slot_bank_size <= 0:
            return
        assert self._slot_bank_lookup is not None
        self._slot_bank_lookup[:] = -1
        self._slot_bank_owner = [-1] * self._slot_bank_size
        self._slot_bank_expert_to_slot.clear()
        self._slot_bank_lru.clear()

    def reset_slot_bank_stats(self) -> None:
        self._slot_bank_requests = 0
        self._slot_bank_misses = 0
        self._slot_bank_full_hit_calls = 0
        self._slot_bank_calls = 0
        self._last_slot_bank_expert_ids = ()
        self._slot_bank_stage_stats.reset()

    def prime_slot_bank(self, expert_ids: Sequence[int]) -> None:
        if self._slot_bank_executor is None:
            raise RuntimeError("Slot-bank mode is not configured")
        unique_expert_ids = list(dict.fromkeys(int(expert_id) for expert_id in expert_ids))
        if len(unique_expert_ids) > self._slot_bank_size:
            raise ValueError(
                f"Cannot prime {len(unique_expert_ids)} experts into a slot bank of size {self._slot_bank_size}"
            )
        self.reset_slot_bank_state()
        if not unique_expert_ids:
            return
        slot_ids = list(range(len(unique_expert_ids)))
        self._install_slot_bank_experts(slot_ids, unique_expert_ids)

    def prefetch_slot_bank(self, expert_ids: Sequence[int]) -> None:
        if self._slot_bank_executor is None:
            raise RuntimeError("Slot-bank mode is not configured")
        unique_expert_ids = list(dict.fromkeys(int(expert_id) for expert_id in expert_ids))
        if not unique_expert_ids:
            return

        if isinstance(self._expert_loader, NativeSlotBankLoader):
            load_result = self._expert_loader.slot_bank_load(self.layer_index, unique_expert_ids)
            if load_result.miss_slot_ids:
                self._slot_bank_executor.load_quantized_views_into_slots(
                    load_result.miss_slot_ids,
                    load_result.miss_views,
                )
            return

        protected_slots: set[int] = set()
        missing_experts: list[int] = []
        for expert_id in unique_expert_ids:
            slot_id = self._slot_bank_expert_to_slot.get(expert_id)
            if slot_id is None:
                missing_experts.append(expert_id)
                continue
            protected_slots.add(slot_id)
            self._touch_slot_bank_slot(slot_id)

        if not missing_experts:
            return

        reserved_slots: set[int] = set()
        chosen_slots: list[int] = []
        for _ in missing_experts:
            slot_id = self._choose_slot_bank_victim(protected_slots, reserved_slots)
            reserved_slots.add(slot_id)
            chosen_slots.append(slot_id)
        self._install_slot_bank_experts(chosen_slots, missing_experts)

    def slot_bank_resident_buffers(self) -> tuple[mx.array, ...]:
        if self._slot_bank_executor is None:
            return ()
        if self._slot_bank_lookup is None:
            return self._slot_bank_executor.resident_buffers()
        return self._slot_bank_executor.resident_buffers() + (self._slot_bank_lookup,)

    def _choose_slot_bank_victim(self, protected_slots: set[int], reserved_slots: set[int]) -> int:
        for slot_id, owner in enumerate(self._slot_bank_owner):
            if owner == -1 and slot_id not in reserved_slots:
                return slot_id
        for slot_id in list(self._slot_bank_lru.keys()):
            if slot_id in protected_slots or slot_id in reserved_slots:
                continue
            return slot_id
        raise RuntimeError("Could not find a slot-bank victim that is not protected")

    def _install_slot_bank_experts(self, slot_ids: list[int], expert_ids: list[int]) -> None:
        if self._slot_bank_executor is None:
            raise RuntimeError("Slot-bank executor is not configured")
        if self._resident_packed_store is not None:
            self._slot_bank_executor.copy_experts_into_slots(self._resident_packed_store, slot_ids, expert_ids)
        else:
            if self._expert_loader is None:
                raise RuntimeError("Slot-bank mode requires an expert loader or resident packed store")
            experts = self._load_expert_views(expert_ids)
            self._slot_bank_executor.load_quantized_views_into_slots(slot_ids, experts)

        for slot_id, expert_id in zip(slot_ids, expert_ids):
            previous_owner = self._slot_bank_owner[slot_id]
            if previous_owner != -1:
                self._slot_bank_expert_to_slot.pop(previous_owner, None)
                assert self._slot_bank_lookup is not None
                self._slot_bank_lookup[previous_owner] = -1
            self._slot_bank_owner[slot_id] = expert_id
            self._slot_bank_expert_to_slot[expert_id] = slot_id
            assert self._slot_bank_lookup is not None
            self._slot_bank_lookup[expert_id] = slot_id
            self._touch_slot_bank_slot(slot_id)

    def _slot_bank_switch_mlp(
        self,
        x: mx.array,
        expert_indices: mx.array,
        expert_scores: mx.array,
    ) -> mx.array:
        if self._slot_bank_executor is None:
            raise RuntimeError("Slot-bank mode is not configured")
        if x.shape[0] != 1 or x.shape[1] != 1:
            raise NotImplementedError(
                "Slot-bank mode currently expects single-token execution so each token can "
                "reuse stable per-layer slot buffers."
            )

        timing_enabled = self._slot_bank_stage_timing_enabled

        if self._slot_bank_direct_contiguous_hit_enabled:
            self._slot_bank_calls += 1
            self._slot_bank_requests += int(np.prod(expert_indices.shape))
            self._slot_bank_full_hit_calls += 1
            if timing_enabled:
                self._slot_bank_stage_stats.calls += 1
                self._slot_bank_stage_stats.hit_calls += 1
            index_start = time.perf_counter()
            local_indices = mx.broadcast_to(mx.arange(self.top_k, dtype=mx.int32), expert_indices.shape)
            if timing_enabled:
                mx.eval(local_indices)
                self._slot_bank_stage_stats.index_seconds += time.perf_counter() - index_start
            forward_start = time.perf_counter()
            routed_y = self._slot_bank_executor(x, local_indices)
            if timing_enabled:
                mx.eval(routed_y)
                self._slot_bank_stage_stats.forward_seconds += time.perf_counter() - forward_start
            combine_start = time.perf_counter()
            combined = (routed_y * expert_scores[..., None]).sum(axis=-2)
            if timing_enabled:
                mx.eval(combined)
                self._slot_bank_stage_stats.combine_seconds += time.perf_counter() - combine_start
            return combined

        if self._slot_bank_device_hit_lookup_enabled:
            assert self._slot_bank_lookup is not None
            self._slot_bank_calls += 1
            self._slot_bank_requests += int(np.prod(expert_indices.shape))
            self._slot_bank_full_hit_calls += 1
            if timing_enabled:
                self._slot_bank_stage_stats.calls += 1
                self._slot_bank_stage_stats.hit_calls += 1
            index_start = time.perf_counter()
            local_indices = mx.take(self._slot_bank_lookup, expert_indices, axis=0)
            if timing_enabled:
                mx.eval(local_indices)
                self._slot_bank_stage_stats.index_seconds += time.perf_counter() - index_start
            forward_start = time.perf_counter()
            routed_y = self._slot_bank_executor(x, local_indices)
            if timing_enabled:
                mx.eval(routed_y)
                self._slot_bank_stage_stats.forward_seconds += time.perf_counter() - forward_start
            combine_start = time.perf_counter()
            combined = (routed_y * expert_scores[..., None]).sum(axis=-2)
            if timing_enabled:
                mx.eval(combined)
                self._slot_bank_stage_stats.combine_seconds += time.perf_counter() - combine_start
            return combined

        selector_start = time.perf_counter()
        token_experts = np.asarray(expert_indices.reshape(-1, self.top_k)[0], dtype=np.int32).tolist()
        self._last_slot_bank_expert_ids = tuple(int(expert_id) for expert_id in token_experts)
        if timing_enabled:
            self._slot_bank_stage_stats.selector_seconds += time.perf_counter() - selector_start

        if isinstance(self._expert_loader, NativeSlotBankLoader):
            lookup_start = time.perf_counter()
            load_result = self._expert_loader.slot_bank_load(self.layer_index, token_experts)
            if timing_enabled:
                self._slot_bank_stage_stats.lookup_seconds += time.perf_counter() - lookup_start
            self._slot_bank_calls += 1
            self._slot_bank_requests += len(token_experts)
            self._slot_bank_misses += len(load_result.miss_expert_ids)
            if not load_result.miss_expert_ids:
                self._slot_bank_full_hit_calls += 1
                if timing_enabled:
                    self._slot_bank_stage_stats.hit_calls += 1
            elif load_result.miss_slot_ids:
                install_start = time.perf_counter()
                self._slot_bank_executor.load_quantized_views_into_slots(
                    load_result.miss_slot_ids,
                    load_result.miss_views,
                )
                if timing_enabled:
                    mx.eval(*self._slot_bank_executor.resident_buffers())
                    self._slot_bank_stage_stats.install_seconds += time.perf_counter() - install_start
                    self._slot_bank_stage_stats.miss_calls += 1
            elif timing_enabled:
                self._slot_bank_stage_stats.miss_calls += 1
            index_start = time.perf_counter()
            local_indices = mx.array(
                np.asarray(load_result.slot_ids, dtype=np.int32).reshape(expert_indices.shape),
                dtype=mx.int32,
            )
            if timing_enabled:
                mx.eval(local_indices)
                self._slot_bank_stage_stats.index_seconds += time.perf_counter() - index_start
            forward_start = time.perf_counter()
            routed_y = self._slot_bank_executor(x, local_indices)
            if timing_enabled:
                mx.eval(routed_y)
                self._slot_bank_stage_stats.forward_seconds += time.perf_counter() - forward_start
            combine_start = time.perf_counter()
            combined = (routed_y * expert_scores[..., None]).sum(axis=-2)
            if timing_enabled:
                mx.eval(combined)
                self._slot_bank_stage_stats.combine_seconds += time.perf_counter() - combine_start
                self._slot_bank_stage_stats.calls += 1
            return combined

        slot_ids = [-1] * len(token_experts)
        protected_slots: set[int] = set()
        missing_experts: list[int] = []
        missing_positions: list[int] = []

        self._slot_bank_calls += 1
        self._slot_bank_requests += len(token_experts)

        lookup_start = time.perf_counter()
        for position, expert_id in enumerate(token_experts):
            slot_id = self._slot_bank_expert_to_slot.get(expert_id)
            if slot_id is None:
                missing_experts.append(expert_id)
                missing_positions.append(position)
                continue
            slot_ids[position] = slot_id
            protected_slots.add(slot_id)
            self._touch_slot_bank_slot(slot_id)
        if timing_enabled:
            self._slot_bank_stage_stats.lookup_seconds += time.perf_counter() - lookup_start

        if not missing_experts:
            self._slot_bank_full_hit_calls += 1
            if timing_enabled:
                self._slot_bank_stage_stats.hit_calls += 1
        else:
            reserved_slots: set[int] = set()
            chosen_slots: list[int] = []
            for _ in missing_experts:
                slot_id = self._choose_slot_bank_victim(protected_slots, reserved_slots)
                reserved_slots.add(slot_id)
                chosen_slots.append(slot_id)
            install_start = time.perf_counter()
            self._install_slot_bank_experts(chosen_slots, missing_experts)
            if timing_enabled:
                mx.eval(*self._slot_bank_executor.resident_buffers())
                self._slot_bank_stage_stats.install_seconds += time.perf_counter() - install_start
                self._slot_bank_stage_stats.miss_calls += 1
            self._slot_bank_misses += len(missing_experts)
            for position, slot_id in zip(missing_positions, chosen_slots):
                slot_ids[position] = slot_id

        index_start = time.perf_counter()
        local_indices = mx.array(np.asarray(slot_ids, dtype=np.int32).reshape(expert_indices.shape), dtype=mx.int32)
        if timing_enabled:
            mx.eval(local_indices)
            self._slot_bank_stage_stats.index_seconds += time.perf_counter() - index_start
        forward_start = time.perf_counter()
        routed_y = self._slot_bank_executor(x, local_indices)
        if timing_enabled:
            mx.eval(routed_y)
            self._slot_bank_stage_stats.forward_seconds += time.perf_counter() - forward_start
        combine_start = time.perf_counter()
        combined = (routed_y * expert_scores[..., None]).sum(axis=-2)
        if timing_enabled:
            mx.eval(combined)
            self._slot_bank_stage_stats.combine_seconds += time.perf_counter() - combine_start
            self._slot_bank_stage_stats.calls += 1
        return combined

    def _streamed_switch_mlp(self, x: mx.array, expert_indices: mx.array, expert_scores: mx.array) -> mx.array:
        if self._expert_loader is None:
            raise RuntimeError("Streamed MoE requested without a native expert loader")
        if x.shape[0] != 1 or x.shape[1] != 1:
            raise NotImplementedError(
                "Streamed SSD MoE currently expects single-token execution so it can "
                "reuse the resident SwitchGLU path with one routed expert set."
            )
        if self._streamed_switch_executor is None:
            raise RuntimeError("Missing streamed switch executor")

        token_experts = np.asarray(expert_indices.reshape(-1, self.top_k)[0], dtype=np.int32).tolist()
        experts = self._load_expert_views(token_experts)

        self._streamed_switch_executor.load_quantized_views(experts)

        local_indices = mx.broadcast_to(mx.arange(len(experts), dtype=mx.int32), expert_indices.shape)
        routed_y = self._streamed_switch_executor(x, local_indices)
        return (routed_y * expert_scores[..., None]).sum(axis=-2)

    def _apply_shared_expert_tail(self, x: mx.array, routed_y: mx.array) -> mx.array:
        shared_y = self.shared_expert(x)
        shared_gate_logits = self.shared_expert_gate(x)
        if not self._compiled_tail_enabled:
            return _apply_sparse_moe_tail(routed_y, shared_gate_logits, shared_y)
        if self._compiled_tail_fn is None:
            self._compiled_tail_fn = mx.compile(
                lambda in_routed_y, in_shared_gate_logits, in_shared_y: _apply_sparse_moe_tail(
                    in_routed_y,
                    in_shared_gate_logits,
                    in_shared_y,
                )
            )
        return self._compiled_tail_fn(routed_y, shared_gate_logits, shared_y)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        expert_indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        expert_scores = mx.take_along_axis(gates, expert_indices, axis=-1)
        expert_order = mx.argsort(-expert_scores, axis=-1)
        expert_indices = mx.take_along_axis(expert_indices, expert_order, axis=-1)
        expert_scores = mx.take_along_axis(expert_scores, expert_order, axis=-1)
        if self.norm_topk_prob:
            expert_scores = expert_scores / expert_scores.sum(axis=-1, keepdims=True)
        if self._trace_callback is not None:
            traced_indices = expert_indices.astype(mx.int32)
            traced_scores = expert_scores.astype(mx.float32)
            self._trace_callback(
                self.layer_index,
                np.asarray(traced_indices).astype(np.int32, copy=False),
                np.asarray(traced_scores).astype(np.float32, copy=False),
            )
        if self._routing_sample_callback is not None:
            self._routing_sample_callback(
                self.layer_index,
                np.asarray(x.astype(mx.float32)).reshape(-1).astype(np.float32, copy=False),
                np.asarray(expert_indices.astype(mx.int32)).reshape(-1).astype(np.int32, copy=False),
            )
        if self._candidate_trace_callback is not None and self._candidate_trace_top_n > 0:
            candidate_top_n = max(self.top_k, min(self._candidate_trace_top_n, gates.shape[-1]))
            if candidate_top_n == self.top_k:
                candidate_indices = expert_indices
                candidate_scores = expert_scores
            else:
                candidate_indices = mx.argpartition(gates, kth=-candidate_top_n, axis=-1)[
                    ..., -candidate_top_n :
                ]
                candidate_scores = mx.take_along_axis(gates, candidate_indices, axis=-1)
                candidate_order = mx.argsort(-candidate_scores, axis=-1)
                candidate_indices = mx.take_along_axis(candidate_indices, candidate_order, axis=-1)
                candidate_scores = mx.take_along_axis(candidate_scores, candidate_order, axis=-1)
            self._candidate_trace_callback(
                self.layer_index,
                np.asarray(candidate_indices.astype(mx.int32)).astype(np.int32, copy=False),
                np.asarray(candidate_scores.astype(mx.float32)).astype(np.float32, copy=False),
            )

        mode_override = self._execution_mode_override

        if self._bypass_routed_mlp:
            routed_y = mx.zeros_like(x)
        elif mode_override == "resident_pread":
            assert self._resident_packed_store is not None
            routed_y = self._bank_index_switch_mlp(
                self._resident_packed_store,
                x,
                expert_indices,
                expert_scores,
            )
        elif mode_override == "resident_rebind":
            routed_y = self._resident_rebind_switch_mlp(x, expert_indices, expert_scores)
        elif mode_override == "resident_copy_k":
            routed_y = self._resident_copy_k_switch_mlp(x, expert_indices, expert_scores)
        elif self._use_resident_experts:
            assert self.switch_mlp is not None
            routed_y = self.switch_mlp(x, expert_indices)
            routed_y = (routed_y * expert_scores[..., None]).sum(axis=-2)
        elif self._use_resident_pread_mlx and self._slot_bank_size <= 0:
            assert self._resident_packed_store is not None
            routed_y = self._bank_index_switch_mlp(
                self._resident_packed_store,
                x,
                expert_indices,
                expert_scores,
            )
        elif self._slot_bank_size > 0:
            routed_y = self._slot_bank_switch_mlp(x, expert_indices, expert_scores)
        elif self._use_resident_rebind:
            routed_y = self._resident_rebind_switch_mlp(x, expert_indices, expert_scores)
        elif self._use_resident_copy_k:
            routed_y = self._resident_copy_k_switch_mlp(x, expert_indices, expert_scores)
        else:
            routed_y = self._streamed_switch_mlp(x, expert_indices, expert_scores)
        return self._apply_shared_expert_tail(x, routed_y)


class Qwen3TextDecoderLayer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_index: int,
        expert_loader: Optional[ExpertLoaderProtocol],
        expert_geometry: ExpertGeometry,
        routed_top_k: int,
        use_resident_experts: bool,
        use_resident_pread_mlx: bool,
        use_resident_rebind: bool,
        use_resident_copy_k: bool,
        slot_bank_size: int,
        expert_bits: int,
        expert_mode: str,
    ):
        super().__init__()
        layer_type = args.layer_types[layer_index] if args.layer_types else None
        self.is_linear = layer_type == "linear_attention" if layer_type else (
            (layer_index + 1) % args.full_attention_interval != 0
        )
        if self.is_linear:
            self.linear_attn = Qwen3TextGatedDeltaNet(args)
        else:
            self.self_attn = Qwen3TextAttention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        is_sparse = (
            layer_index not in args.mlp_only_layers
            and args.num_experts > 0
            and (layer_index + 1) % args.decoder_sparse_step == 0
        )
        if not is_sparse:
            self.mlp = Qwen3TextMLP(args, args.intermediate_size)
        else:
            self.mlp = Qwen3TextSparseMoeBlock(
                args=args,
                layer_index=layer_index,
                expert_loader=expert_loader,
                expert_geometry=expert_geometry,
                routed_top_k=routed_top_k,
                use_resident_experts=use_resident_experts,
                use_resident_pread_mlx=use_resident_pread_mlx,
                use_resident_rebind=use_resident_rebind,
                use_resident_copy_k=use_resident_copy_k,
                slot_bank_size=slot_bank_size,
                expert_bits=expert_bits,
                expert_mode=expert_mode,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if self.is_linear:
            residual = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            residual = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        hidden = x + residual
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen3TextModel(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        expert_loader: Optional[ExpertLoaderProtocol],
        expert_geometry: ExpertGeometry,
        routed_top_k: int,
        use_resident_experts: bool,
        use_resident_pread_mlx: bool,
        use_resident_rebind: bool,
        use_resident_copy_k: bool,
        slot_bank_size: int,
        expert_bits: int,
        expert_mode: str,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3TextDecoderLayer(
                args=args,
                layer_index=layer_index,
                expert_loader=expert_loader,
                expert_geometry=expert_geometry,
                routed_top_k=routed_top_k,
                use_resident_experts=use_resident_experts,
                use_resident_pread_mlx=use_resident_pread_mlx,
                use_resident_rebind=use_resident_rebind,
                use_resident_copy_k=use_resident_copy_k,
                slot_bank_size=slot_bank_size,
                expert_bits=expert_bits,
                expert_mode=expert_mode,
            )
            for layer_index in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = next(
            (index for index, layer in enumerate(self.layers) if layer.is_linear),
            0,
        )
        self.fa_idx = next(
            (index for index, layer in enumerate(self.layers) if not layer.is_linear),
            0,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[list[Any]] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        full_attn_mask = create_attention_mask(
            hidden_states,
            cache[self.fa_idx] if cache else None,
        )
        ssm_mask = create_ssm_mask(
            hidden_states,
            cache[self.ssm_idx] if cache else None,
        )
        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else full_attn_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache, position_ids=position_ids)
        return self.norm(hidden_states)


class Qwen3LanguageModel(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        expert_loader: Optional[ExpertLoaderProtocol],
        expert_geometry: ExpertGeometry,
        routed_top_k: int,
        use_resident_experts: bool,
        use_resident_pread_mlx: bool,
        use_resident_rebind: bool,
        use_resident_copy_k: bool,
        slot_bank_size: int,
        expert_bits: int,
        expert_mode: str,
    ):
        super().__init__()
        self.model = Qwen3TextModel(
            args=args,
            expert_loader=expert_loader,
            expert_geometry=expert_geometry,
            routed_top_k=routed_top_k,
            use_resident_experts=use_resident_experts,
            use_resident_pread_mlx=use_resident_pread_mlx,
            use_resident_rebind=use_resident_rebind,
            use_resident_copy_k=use_resident_copy_k,
            slot_bank_size=slot_bank_size,
            expert_bits=expert_bits,
            expert_mode=expert_mode,
        )
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)


class TextOnlyQwen35(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        expert_loader: Optional[ExpertLoaderProtocol],
        expert_geometry: ExpertGeometry,
        routed_top_k: int,
        use_resident_experts: bool,
        use_resident_pread_mlx: bool,
        use_resident_rebind: bool,
        use_resident_copy_k: bool,
        slot_bank_size: int,
        expert_bits: int,
        expert_mode: str,
    ):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = Qwen3LanguageModel(
            args=args,
            expert_loader=expert_loader,
            expert_geometry=expert_geometry,
            routed_top_k=routed_top_k,
            use_resident_experts=use_resident_experts,
            use_resident_pread_mlx=use_resident_pread_mlx,
            use_resident_rebind=use_resident_rebind,
            use_resident_copy_k=use_resident_copy_k,
            slot_bank_size=slot_bank_size,
            expert_bits=expert_bits,
            expert_mode=expert_mode,
        )
        self.use_resident_experts = use_resident_experts
        self.use_resident_pread_mlx = use_resident_pread_mlx
        self.use_resident_rebind = use_resident_rebind
        self.use_resident_copy_k = use_resident_copy_k
        self.slot_bank_size = slot_bank_size
        self.expert_bits = expert_bits
        self.expert_mode = expert_mode
        self._position_ids = None
        self._rope_deltas = None

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return [MambaCache() if layer.is_linear else KVCache() for layer in self.layers]

    def get_rope_index(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        if attention_mask is not None:
            position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            position_ids = mx.where(
                attention_mask == 0,
                mx.ones_like(position_ids),
                position_ids,
            )
            position_ids = mx.expand_dims(position_ids[0], axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
            max_position_ids = position_ids.max(0, keepdims=False)[0].max(-1, keepdims=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
            position_ids = mx.broadcast_to(
                position_ids,
                (3, input_ids.shape[0], input_ids.shape[1]),
            )
            mrope_position_deltas = mx.zeros(
                [input_ids.shape[0], 1],
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[list[Any]] = None,
        mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        cache_offset = 0
        if cache and cache[self.language_model.model.fa_idx] is not None:
            offset = cache[self.language_model.model.fa_idx].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        rope_mask = mask
        if mask is not None and mask.shape[-1] != inputs.shape[-1]:
            rope_mask = None

        if position_ids is None and (rope_mask is None or rope_mask.ndim == 2):
            if (
                (
                    cache is not None
                    and cache[self.language_model.model.fa_idx] is not None
                    and cache_offset == 0
                )
                or self._rope_deltas is None
                or cache is None
            ):
                if self._position_ids is not None:
                    seq_length = inputs.shape[1]
                    position_ids = self._position_ids[
                        :,
                        :,
                        cache_offset : cache_offset + seq_length,
                    ]
                else:
                    position_ids, rope_deltas = self.get_rope_index(inputs, rope_mask)
                    self._rope_deltas = rope_deltas
                    self._position_ids = position_ids
            else:
                batch_size, seq_length = inputs.shape
                delta = mx.array(
                    cache_offset + self._rope_deltas if cache is not None else 0
                )
                position_ids = mx.arange(seq_length).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

                if delta.ndim == 0:
                    delta = mx.expand_dims(delta, axis=0)
                if delta.shape[0] < batch_size:
                    delta = mx.tile(delta, (batch_size, 1))
                else:
                    delta = delta[:batch_size]

                position_ids = mx.add(position_ids, delta)[None, ...]
                position_ids = mx.broadcast_to(
                    position_ids,
                    (3, batch_size, seq_length),
                )

        hidden = self.language_model.model(inputs, cache=cache, position_ids=position_ids)
        if self.args.tie_word_embeddings:
            return self.language_model.model.embed_tokens.as_linear(hidden)
        return self.language_model.lm_head(hidden)


def _iter_weight_files(model_dir: Path) -> Iterator[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        seen = set()
        for shard_name in weight_map.values():
            if shard_name not in seen:
                seen.add(shard_name)
                yield model_dir / shard_name
        return
    yield from sorted(model_dir.glob("*.safetensors"))


def validate_expert_directory(
    experts_dir: Path,
    num_layers: int,
    expected_layer_size: int,
) -> list[Path]:
    if not experts_dir.is_dir():
        raise FileNotFoundError(f"Expert directory does not exist: {experts_dir}")
    layer_paths = [experts_dir / f"layer_{layer:02d}.bin" for layer in range(num_layers)]
    missing = [path for path in layer_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expert layer files: {', '.join(str(path) for path in missing[:4])}")
    for path in layer_paths:
        size = path.stat().st_size
        if size != expected_layer_size:
            raise ValueError(
                f"Unexpected size for {path.name}: expected {expected_layer_size} bytes, got {size}"
            )
    return layer_paths


def _load_text_weights(model_dir: Path, include_resident_experts: bool) -> dict[str, mx.array]:
    filtered_weights: dict[str, mx.array] = {}
    for weight_file in _iter_weight_files(model_dir):
        arrays = mx.load(weight_file)
        for key, value in arrays.items():
            if not key.startswith("language_model."):
                continue
            if ".mlp.switch_mlp." in key and not include_resident_experts:
                continue
            filtered_weights[key] = value
    return filtered_weights


def _quantize_text_model(
    model: TextOnlyQwen35,
    args: ModelArgs,
    weights: dict[str, mx.array],
) -> None:
    quant_overrides = args.quantization_overrides

    if not quant_overrides:
        def class_predicate(path: str, module: nn.Module):
            if not hasattr(module, "to_quantized"):
                return False
            weight = getattr(module, "weight", None)
            if weight is not None and weight.size % args.quant_group_size != 0:
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=args.quant_group_size,
            bits=args.quant_bits,
            mode=args.quant_mode,
            class_predicate=class_predicate,
        )
        return

    def class_predicate(path: str, module: nn.Module):
        if not hasattr(module, "to_quantized"):
            return False
        weight = getattr(module, "weight", None)
        if weight is not None and weight.size % args.quant_group_size != 0:
            return False
        if f"{path}.scales" not in weights:
            return False
        override = quant_overrides.get(path)
        if override is None:
            return True
        if (
            int(override.get("bits", args.quant_bits)) == args.quant_bits
            and int(override.get("group_size", args.quant_group_size)) == args.quant_group_size
            and str(override.get("mode", args.quant_mode)) == args.quant_mode
        ):
            return True
        return {
            "group_size": int(override.get("group_size", args.quant_group_size)),
            "bits": int(override.get("bits", args.quant_bits)),
            "mode": str(override.get("mode", args.quant_mode)),
        }

    nn.quantize(
        model,
        group_size=args.quant_group_size,
        bits=args.quant_bits,
        mode=args.quant_mode,
        class_predicate=class_predicate,
    )


def load_language_model_weights(model_dir: Path, model: TextOnlyQwen35) -> None:
    filtered_weights = _load_text_weights(
        model_dir,
        include_resident_experts=model.use_resident_experts,
    )
    _quantize_text_model(model, model.args, filtered_weights)

    param_keys = set(tree_flatten(model.parameters(), destination={}).keys())
    found_keys = set()
    loadable_weights = []
    for key, value in filtered_weights.items():
        if key in param_keys:
            loadable_weights.append((key, value))
            found_keys.add(key)
    missing = sorted(param_keys - found_keys)
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"Missing {len(missing)} required text weights, starting with: {preview}")
    model.load_weights(loadable_weights, strict=False)


def _iter_sparse_moe_blocks(model: TextOnlyQwen35) -> Iterator[Qwen3TextSparseMoeBlock]:
    for layer in model.layers:
        if isinstance(layer.mlp, Qwen3TextSparseMoeBlock):
            yield layer.mlp


def _set_bypass_routed_mlp(model: TextOnlyQwen35, enabled: bool) -> None:
    for moe_block in _iter_sparse_moe_blocks(model):
        moe_block._bypass_routed_mlp = enabled


def _set_routing_trace_callback(
    model: TextOnlyQwen35,
    callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]],
) -> None:
    for moe_block in _iter_sparse_moe_blocks(model):
        moe_block._trace_callback = callback


def set_routing_sample_callback(
    model: TextOnlyQwen35,
    callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]],
) -> None:
    for moe_block in _iter_sparse_moe_blocks(model):
        moe_block._routing_sample_callback = callback


def _set_routing_candidate_trace_callback(
    model: TextOnlyQwen35,
    callback: Optional[Callable[[int, np.ndarray, np.ndarray], None]],
    top_n: int,
) -> None:
    trace_top_n = max(0, int(top_n))
    for moe_block in _iter_sparse_moe_blocks(model):
        moe_block._candidate_trace_callback = callback
        moe_block._candidate_trace_top_n = trace_top_n


def list_sparse_moe_layer_indices(model: TextOnlyQwen35) -> list[int]:
    return [moe_block.layer_index for moe_block in _iter_sparse_moe_blocks(model)]


def set_sparse_moe_mode_overrides(
    model: TextOnlyQwen35,
    mode_by_layer: Optional[Dict[int, str]],
) -> None:
    allowed = {None, "resident_pread", "resident_rebind", "resident_copy_k"}
    for moe_block in _iter_sparse_moe_blocks(model):
        mode = None if mode_by_layer is None else mode_by_layer.get(moe_block.layer_index)
        if mode not in allowed:
            raise ValueError(f"Unsupported sparse MoE override for layer {moe_block.layer_index}: {mode}")
        moe_block._execution_mode_override = mode


def set_bank_index_options(
    model: TextOnlyQwen35,
    *,
    sort_ids: bool,
    use_compile: bool,
    layers: Optional[Sequence[int]] = None,
) -> None:
    selected = None if layers is None else {int(layer) for layer in layers}
    for moe_block in _iter_sparse_moe_blocks(model):
        if selected is not None and moe_block.layer_index not in selected:
            continue
        moe_block._bank_index_sort_ids = sort_ids
        moe_block._bank_index_use_compile = use_compile


def set_sparse_moe_tail_compile(
    model: TextOnlyQwen35,
    enabled: bool,
    layers: Optional[Sequence[int]] = None,
) -> None:
    selected = None if layers is None else {int(layer) for layer in layers}
    for moe_block in _iter_sparse_moe_blocks(model):
        if selected is not None and moe_block.layer_index not in selected:
            continue
        moe_block._compiled_tail_enabled = enabled


def enable_rebind_stage_timing(
    model: TextOnlyQwen35,
    enabled: bool,
    layers: Optional[Sequence[int]] = None,
) -> None:
    selected = None if layers is None else {int(layer) for layer in layers}
    for moe_block in _iter_sparse_moe_blocks(model):
        moe_block._rebind_stage_timing_enabled = (
            enabled and (selected is None or moe_block.layer_index in selected)
        )


def reset_rebind_stage_stats(model: TextOnlyQwen35) -> None:
    for moe_block in _iter_sparse_moe_blocks(model):
        moe_block._rebind_stage_stats.reset()


def collect_rebind_stage_stats(model: TextOnlyQwen35) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    total = RebindStageStats()
    for moe_block in _iter_sparse_moe_blocks(model):
        stats = moe_block._rebind_stage_stats
        if stats.calls <= 0:
            continue
        layer_total = stats.total_seconds
        layers.append(
            {
                "layer": moe_block.layer_index,
                "calls": stats.calls,
                "selector_seconds": stats.selector_seconds,
                "take_seconds": stats.take_seconds,
                "mutate_seconds": stats.mutate_seconds,
                "forward_seconds": stats.forward_seconds,
                "combine_seconds": stats.combine_seconds,
                "total_seconds": layer_total,
            }
        )
        total.calls += stats.calls
        total.selector_seconds += stats.selector_seconds
        total.take_seconds += stats.take_seconds
        total.mutate_seconds += stats.mutate_seconds
        total.forward_seconds += stats.forward_seconds
        total.combine_seconds += stats.combine_seconds
    return {
        "calls": total.calls,
        "selector_seconds": total.selector_seconds,
        "take_seconds": total.take_seconds,
        "mutate_seconds": total.mutate_seconds,
        "forward_seconds": total.forward_seconds,
        "combine_seconds": total.combine_seconds,
        "total_seconds": total.total_seconds,
        "layers": layers,
    }


def collect_slot_bank_stats(model: TextOnlyQwen35) -> dict[str, float]:
    total_calls = 0
    total_requests = 0
    total_misses = 0
    total_full_hit_calls = 0
    active_layers = 0
    for moe_block in _iter_sparse_moe_blocks(model):
        if moe_block._slot_bank_size <= 0:
            continue
        active_layers += 1
        total_calls += moe_block._slot_bank_calls
        total_requests += moe_block._slot_bank_requests
        total_misses += moe_block._slot_bank_misses
        total_full_hit_calls += moe_block._slot_bank_full_hit_calls
    hit_rate = 0.0 if total_requests == 0 else 1.0 - (total_misses / float(total_requests))
    full_hit_rate = 0.0 if total_calls == 0 else total_full_hit_calls / float(total_calls)
    return {
        "layers": float(active_layers),
        "calls": float(total_calls),
        "requests": float(total_requests),
        "misses": float(total_misses),
        "hit_rate": hit_rate,
        "full_hit_rate": full_hit_rate,
    }


def enable_slot_bank_stage_timing(
    model: TextOnlyQwen35,
    enabled: bool,
    layers: Optional[Sequence[int]] = None,
) -> None:
    selected = None if layers is None else {int(layer) for layer in layers}
    for moe_block in _iter_sparse_moe_blocks(model):
        moe_block._slot_bank_stage_timing_enabled = (
            enabled and (selected is None or moe_block.layer_index in selected)
        )


def set_slot_bank_device_hit_lookup(
    model: TextOnlyQwen35,
    enabled: bool,
    layers: Optional[Sequence[int]] = None,
) -> None:
    selected = None if layers is None else {int(layer) for layer in layers}
    for moe_block in _iter_sparse_moe_blocks(model):
        if selected is not None and moe_block.layer_index not in selected:
            continue
        moe_block._slot_bank_device_hit_lookup_enabled = enabled


def set_slot_bank_direct_contiguous_hit(
    model: TextOnlyQwen35,
    enabled: bool,
    layers: Optional[Sequence[int]] = None,
) -> None:
    selected = None if layers is None else {int(layer) for layer in layers}
    for moe_block in _iter_sparse_moe_blocks(model):
        if selected is not None and moe_block.layer_index not in selected:
            continue
        moe_block._slot_bank_direct_contiguous_hit_enabled = enabled


def collect_slot_bank_stage_stats(model: TextOnlyQwen35) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    total = SlotBankStageStats()
    for moe_block in _iter_sparse_moe_blocks(model):
        stats = moe_block._slot_bank_stage_stats
        if stats.calls <= 0:
            continue
        layer_total = stats.total_seconds
        layers.append(
            {
                "layer": moe_block.layer_index,
                "calls": stats.calls,
                "hit_calls": stats.hit_calls,
                "miss_calls": stats.miss_calls,
                "selector_seconds": stats.selector_seconds,
                "lookup_seconds": stats.lookup_seconds,
                "install_seconds": stats.install_seconds,
                "index_seconds": stats.index_seconds,
                "forward_seconds": stats.forward_seconds,
                "combine_seconds": stats.combine_seconds,
                "total_seconds": layer_total,
            }
        )
        total.calls += stats.calls
        total.hit_calls += stats.hit_calls
        total.miss_calls += stats.miss_calls
        total.selector_seconds += stats.selector_seconds
        total.lookup_seconds += stats.lookup_seconds
        total.install_seconds += stats.install_seconds
        total.index_seconds += stats.index_seconds
        total.forward_seconds += stats.forward_seconds
        total.combine_seconds += stats.combine_seconds
    return {
        "calls": total.calls,
        "hit_calls": total.hit_calls,
        "miss_calls": total.miss_calls,
        "selector_seconds": total.selector_seconds,
        "lookup_seconds": total.lookup_seconds,
        "install_seconds": total.install_seconds,
        "index_seconds": total.index_seconds,
        "forward_seconds": total.forward_seconds,
        "combine_seconds": total.combine_seconds,
        "total_seconds": total.total_seconds,
        "layers": layers,
    }


def reset_slot_bank_state(model: TextOnlyQwen35) -> None:
    for moe_block in _iter_sparse_moe_blocks(model):
        moe_block.reset_slot_bank_state()


def reset_slot_bank_stats(model: TextOnlyQwen35) -> None:
    for moe_block in _iter_sparse_moe_blocks(model):
        moe_block.reset_slot_bank_stats()


def prime_slot_banks(model: TextOnlyQwen35, experts_by_layer: Dict[int, Sequence[int]]) -> None:
    for moe_block in _iter_sparse_moe_blocks(model):
        expert_ids = experts_by_layer.get(moe_block.layer_index, ())
        moe_block.prime_slot_bank(expert_ids)


def collect_slot_bank_last_experts(model: TextOnlyQwen35) -> Dict[int, list[int]]:
    experts_by_layer: Dict[int, list[int]] = {}
    for moe_block in _iter_sparse_moe_blocks(model):
        if moe_block._slot_bank_size <= 0 or not moe_block._last_slot_bank_expert_ids:
            continue
        experts_by_layer[moe_block.layer_index] = list(moe_block._last_slot_bank_expert_ids)
    return experts_by_layer


def prefetch_slot_banks(model: TextOnlyQwen35, experts_by_layer: Dict[int, Sequence[int]]) -> None:
    for moe_block in _iter_sparse_moe_blocks(model):
        expert_ids = experts_by_layer.get(moe_block.layer_index, ())
        moe_block.prefetch_slot_bank(expert_ids)


def eval_slot_bank_buffers(model: TextOnlyQwen35) -> None:
    buffers: list[mx.array] = []
    for moe_block in _iter_sparse_moe_blocks(model):
        buffers.extend(moe_block.slot_bank_resident_buffers())
    if buffers:
        mx.eval(*buffers)


def _is_mixed_sidecar_dir(experts_dir: Path) -> bool:
    layout_path = experts_dir / "layout.json"
    if not layout_path.exists():
        return False
    try:
        with layout_path.open("r", encoding="utf-8") as f:
            layout = json.load(f)
    except Exception:
        return False
    return str(layout.get("format", "")) == "mlx-flash-moe-mixed-sidecar-v1"


def _preload_packed_experts_into_mlx(
    model: TextOnlyQwen35,
    experts_dir: Path,
    expert_geometry: ExpertGeometry,
    cache_io_split: int,
) -> None:
    all_expert_ids = list(range(model.args.num_experts))
    if _is_mixed_sidecar_dir(experts_dir):
        preload_loader = MixedPrecisionSidecarLoader(
            experts_dir=experts_dir,
            num_layers=model.args.num_hidden_layers,
            max_k=model.args.num_experts,
        )
    else:
        preload_loader = NativeExpertLoader(
            experts_dir=experts_dir,
            num_layers=model.args.num_hidden_layers,
            geometry=expert_geometry,
            max_k=model.args.num_experts,
            cache_io_split=cache_io_split,
        )
    try:
        for moe_block in _iter_sparse_moe_blocks(model):
            expert_views = preload_loader.load_layer_views(moe_block.layer_index, all_expert_ids)
            moe_block.load_packed_resident_experts(expert_views)
            assert moe_block._resident_packed_store is not None
            mx.eval(*moe_block._resident_packed_store.resident_buffers())
    finally:
        preload_loader.close()


def load_model_bundle(
    mlx_model_dir: Path | str,
    experts_dir: Path | str | None,
    routed_top_k: Optional[int] = None,
    cache_io_split: int = 1,
    expert_bits: Optional[int] = None,
    expert_mode: Optional[str] = None,
    resident_experts: bool = False,
    resident_flash: bool = False,
    resident_pread_mlx: bool = False,
    resident_rebind: bool = False,
    resident_copy_k: bool = False,
    slot_bank_size: int = 0,
    slot_bank_native: bool = False,
    bypass_routed_mlp: bool = False,
) -> ModelBundle:
    mlx_model_dir = Path(mlx_model_dir)
    experts_dir = Path(experts_dir) if experts_dir is not None else None
    mixed_sidecar = bool(experts_dir and _is_mixed_sidecar_dir(experts_dir))
    resident_modes = sum(
        (resident_experts, resident_flash, resident_pread_mlx, resident_rebind, resident_copy_k)
    )
    if resident_modes > 1:
        raise ValueError(
            "--resident, --resident-flash, --resident-pread-mlx, --resident-rebind, and --resident-copy-k are mutually exclusive"
        )
    args = ModelArgs.from_model_dir(mlx_model_dir)
    effective_k = routed_top_k or args.num_experts_per_tok
    if effective_k < 1 or effective_k > args.num_experts_per_tok:
        raise ValueError(
            f"--k must be in the range 1..{args.num_experts_per_tok}, received {effective_k}"
        )
    if slot_bank_size < 0:
        raise ValueError("--slot-bank must be >= 0")
    if slot_bank_size and slot_bank_size < effective_k:
        raise ValueError(f"--slot-bank must be >= --k ({effective_k}), received {slot_bank_size}")
    if slot_bank_size and (resident_experts or resident_flash or resident_rebind or resident_copy_k):
        raise ValueError(
            "--slot-bank can only be combined with the packed resident bank mode (--resident-pread-mlx)"
        )
    effective_expert_bits = args.quant_bits if expert_bits is None else int(expert_bits)
    effective_expert_mode = args.quant_mode if expert_mode is None else str(expert_mode)

    expert_geometry = ExpertGeometry(
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        num_experts=args.num_experts,
        group_size=args.quant_group_size,
        bits=effective_expert_bits,
        mode=effective_expert_mode,
    )
    expert_geometry.validate()
    expert_loader: Optional[ExpertLoaderProtocol] = None
    if not resident_experts:
        if experts_dir is None:
            raise ValueError("--experts is required unless --resident is set")
        if mixed_sidecar:
            if resident_flash:
                raise ValueError("--resident-flash is not supported with mixed sidecar experts yet")
            expert_loader = MixedPrecisionSidecarLoader(
                experts_dir=experts_dir,
                num_layers=args.num_hidden_layers,
                max_k=args.num_experts_per_tok,
            )
        else:
            validate_expert_directory(
                experts_dir,
                num_layers=args.num_hidden_layers,
                expected_layer_size=expert_geometry.layer_file_size,
            )
        if resident_flash:
            expert_loader = ResidentFlashExpertLoader(
                experts_dir=experts_dir,
                num_layers=args.num_hidden_layers,
                geometry=expert_geometry,
                max_k=args.num_experts_per_tok,
            )
        elif not resident_pread_mlx and not resident_rebind and not resident_copy_k and not mixed_sidecar:
            if slot_bank_size and slot_bank_native:
                expert_loader = NativeSlotBankLoader(
                    experts_dir=experts_dir,
                    num_layers=args.num_hidden_layers,
                    geometry=expert_geometry,
                    slot_bank_size=slot_bank_size,
                    cache_io_split=cache_io_split,
                )
            else:
                expert_loader = NativeExpertLoader(
                    experts_dir=experts_dir,
                    num_layers=args.num_hidden_layers,
                    geometry=expert_geometry,
                    max_k=args.num_experts_per_tok,
                    cache_io_split=cache_io_split,
                )
    model = TextOnlyQwen35(
        args=args,
        expert_loader=expert_loader,
        expert_geometry=expert_geometry,
        routed_top_k=effective_k,
        use_resident_experts=resident_experts,
        use_resident_pread_mlx=resident_pread_mlx,
        use_resident_rebind=resident_rebind,
        use_resident_copy_k=resident_copy_k,
        slot_bank_size=slot_bank_size,
        expert_bits=effective_expert_bits,
        expert_mode=effective_expert_mode,
    )
    load_language_model_weights(mlx_model_dir, model)
    if resident_pread_mlx or resident_rebind or resident_copy_k:
        assert experts_dir is not None
        _preload_packed_experts_into_mlx(
            model=model,
            experts_dir=experts_dir,
            expert_geometry=expert_geometry,
            cache_io_split=cache_io_split,
        )
    _set_bypass_routed_mlp(model, bypass_routed_mlp)
    model.eval()
    tokenizer = Tokenizer.from_file(str(mlx_model_dir / "tokenizer.json"))
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        config=args,
        expert_loader=expert_loader,
        expert_geometry=expert_geometry,
        expert_bits=effective_expert_bits,
        expert_mode=effective_expert_mode,
        resident_pread_mlx=resident_pread_mlx,
        resident_rebind=resident_rebind,
        resident_copy_k=resident_copy_k,
        slot_bank_size=slot_bank_size,
        slot_bank_native=slot_bank_native,
        bypass_routed_mlp=bypass_routed_mlp,
    )


def generate_token_ids(
    bundle: ModelBundle,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    seed: Optional[int] = None,
) -> Iterator[int]:
    generated_ids: list[int] = []

    def collect_token(token_id: int) -> None:
        generated_ids.append(token_id)

    generate_with_stats(
        bundle=bundle,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        on_token=collect_token,
    )
    yield from generated_ids


def generate_with_stats(
    bundle: ModelBundle,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    on_token: Optional[Callable[[int], None]] = None,
    slot_bank_temporal_prefetch: bool = False,
) -> GenerationStats:
    encoded = bundle.tokenizer.encode(prompt)
    prompt_ids = list(encoded.ids)
    if not prompt_ids:
        raise ValueError("Prompt tokenized to an empty sequence")

    rng = np.random.default_rng(seed)
    cache = bundle.model.make_cache()
    logits = None
    prefill_start = time.perf_counter()
    for token_id in prompt_ids:
        logits = bundle.model(mx.array([[token_id]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        if slot_bank_temporal_prefetch and bundle.slot_bank_size > 0:
            experts_by_layer = collect_slot_bank_last_experts(bundle.model)
            if experts_by_layer:
                prefetch_slot_banks(bundle.model, experts_by_layer)
                eval_slot_bank_buffers(bundle.model)
    prefill_seconds = time.perf_counter() - prefill_start
    if logits is None:
        return GenerationStats(
            prompt_tokens=len(prompt_ids),
            generated_tokens=0,
            prefill_seconds=prefill_seconds,
            decode_seconds=0.0,
        )

    eos_token_ids = set(bundle.config.eos_token_ids)
    generated_tokens = 0
    decode_start = time.perf_counter()
    for _ in range(max_tokens):
        next_token = _sample_next_token(logits[:, -1, :], temperature, rng)
        if next_token in eos_token_ids:
            break
        generated_tokens += 1
        if on_token is not None:
            on_token(next_token)
        logits = bundle.model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        if slot_bank_temporal_prefetch and bundle.slot_bank_size > 0:
            experts_by_layer = collect_slot_bank_last_experts(bundle.model)
            if experts_by_layer:
                prefetch_slot_banks(bundle.model, experts_by_layer)
                eval_slot_bank_buffers(bundle.model)
    decode_seconds = time.perf_counter() - decode_start
    return GenerationStats(
        prompt_tokens=len(prompt_ids),
        generated_tokens=generated_tokens,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
    )


def decode_incremental(tokenizer: Tokenizer, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(list(token_ids), skip_special_tokens=False)
