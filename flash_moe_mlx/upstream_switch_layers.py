from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


def _gather_sort(x: mx.array, indices: mx.array):
    *_, width = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // width], indices[order], inv_order


def _scatter_unsort(x: mx.array, inv_order: mx.array, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


def _quantized_switch_linear_forward(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    indices: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str,
    sorted_indices: bool,
) -> mx.array:
    return mx.gather_qmm(
        x,
        weight,
        scales,
        biases,
        rhs_indices=indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=sorted_indices,
    )


def quantized_switch_glu_bank_forward(
    x: mx.array,
    indices: mx.array,
    gate_weight: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    up_weight: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array,
    *,
    group_size: int,
    gate_bits: int,
    up_bits: int,
    down_bits: int,
    mode: str,
    assume_sorted_indices: bool = False,
) -> mx.array:
    x = mx.expand_dims(x, (-2, -3))

    do_sort = (not assume_sorted_indices) and indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)

    sorted_hint = assume_sorted_indices or do_sort
    x_up = _quantized_switch_linear_forward(
        x,
        up_weight,
        up_scales,
        up_biases,
        idx,
        group_size=group_size,
        bits=up_bits,
        mode=mode,
        sorted_indices=sorted_hint,
    )
    x_gate = _quantized_switch_linear_forward(
        x,
        gate_weight,
        gate_scales,
        gate_biases,
        idx,
        group_size=group_size,
        bits=gate_bits,
        mode=mode,
        sorted_indices=sorted_hint,
    )
    x = _quantized_switch_linear_forward(
        swiglu(x_gate, x_up),
        down_weight,
        down_scales,
        down_biases,
        idx,
        group_size=group_size,
        bits=down_bits,
        mode=mode,
        sorted_indices=sorted_hint,
    )

    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)

    return x.squeeze(-2)


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        x = _quantized_switch_linear_forward(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            indices,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        quantized = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        quantized.weight, quantized.scales, *biases = mx.quantize(
            self.weight,
            group_size,
            bits,
            mode=mode,
        )
        quantized.biases = biases[0] if biases else None

        if "bias" in self:
            quantized.bias = self.bias
        return quantized


class SwiGLU(nn.Module):
    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class QuantizedSwitchGLUExecutor:
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        group_size: int = 64,
        bits: int = 4,
        gate_bits: int | None = None,
        up_bits: int | None = None,
        down_bits: int | None = None,
        mode: str = "affine",
    ):
        gate_bits = bits if gate_bits is None else gate_bits
        up_bits = bits if up_bits is None else up_bits
        down_bits = bits if down_bits is None else down_bits
        self.gate_proj = QuantizedSwitchLinear(
            input_dims,
            hidden_dims,
            num_experts,
            bias=False,
            group_size=group_size,
            bits=gate_bits,
            mode=mode,
        )
        self.up_proj = QuantizedSwitchLinear(
            input_dims,
            hidden_dims,
            num_experts,
            bias=False,
            group_size=group_size,
            bits=up_bits,
            mode=mode,
        )
        self.down_proj = QuantizedSwitchLinear(
            hidden_dims,
            input_dims,
            num_experts,
            bias=False,
            group_size=group_size,
            bits=down_bits,
            mode=mode,
        )
        gate_scale_shape = self.gate_proj.scales.shape
        up_scale_shape = self.up_proj.scales.shape
        down_scale_shape = self.down_proj.scales.shape

        self._gate_weight = mx.zeros(self.gate_proj.weight.shape, dtype=mx.uint32)
        self._gate_scales_u16 = mx.zeros(gate_scale_shape, dtype=mx.uint16)
        self._gate_biases_u16 = mx.zeros(gate_scale_shape, dtype=mx.uint16)

        self._up_weight = mx.zeros(self.up_proj.weight.shape, dtype=mx.uint32)
        self._up_scales_u16 = mx.zeros(up_scale_shape, dtype=mx.uint16)
        self._up_biases_u16 = mx.zeros(up_scale_shape, dtype=mx.uint16)

        self._down_weight = mx.zeros(self.down_proj.weight.shape, dtype=mx.uint32)
        self._down_scales_u16 = mx.zeros(down_scale_shape, dtype=mx.uint16)
        self._down_biases_u16 = mx.zeros(down_scale_shape, dtype=mx.uint16)

        self.gate_proj.weight = self._gate_weight
        self.up_proj.weight = self._up_weight
        self.down_proj.weight = self._down_weight
        self._refresh_scale_bias_views()
        self._compiled_unsorted_call = None
        self._compiled_sorted_call = None

    def _refresh_scale_bias_views(self) -> None:
        self.gate_proj.scales = mx.view(self._gate_scales_u16, mx.bfloat16)
        self.gate_proj.biases = mx.view(self._gate_biases_u16, mx.bfloat16)
        self.up_proj.scales = mx.view(self._up_scales_u16, mx.bfloat16)
        self.up_proj.biases = mx.view(self._up_biases_u16, mx.bfloat16)
        self.down_proj.scales = mx.view(self._down_scales_u16, mx.bfloat16)
        self.down_proj.biases = mx.view(self._down_biases_u16, mx.bfloat16)

    def load_quantized(
        self,
        *,
        gate_weight: mx.array,
        gate_scales: mx.array,
        gate_biases: mx.array,
        up_weight: mx.array,
        up_scales: mx.array,
        up_biases: mx.array,
        down_weight: mx.array,
        down_scales: mx.array,
        down_biases: mx.array,
    ) -> None:
        self.gate_proj.weight = gate_weight
        self.gate_proj.scales = gate_scales
        self.gate_proj.biases = gate_biases
        self.up_proj.weight = up_weight
        self.up_proj.scales = up_scales
        self.up_proj.biases = up_biases
        self.down_proj.weight = down_weight
        self.down_proj.scales = down_scales
        self.down_proj.biases = down_biases

    def load_quantized_views(self, experts) -> None:
        self.load_quantized_views_into_slots(range(len(experts)), experts)

    def load_quantized_views_into_slots(self, slot_ids, experts) -> None:
        for slot, expert in zip(slot_ids, experts):
            self._gate_weight[slot] = expert.gate_weight
            self._gate_scales_u16[slot] = expert.gate_scales_bf16
            self._gate_biases_u16[slot] = expert.gate_biases_bf16

            self._up_weight[slot] = expert.up_weight
            self._up_scales_u16[slot] = expert.up_scales_bf16
            self._up_biases_u16[slot] = expert.up_biases_bf16

            self._down_weight[slot] = expert.down_weight
            self._down_scales_u16[slot] = expert.down_scales_bf16
            self._down_biases_u16[slot] = expert.down_biases_bf16
        self._refresh_scale_bias_views()

    def copy_experts_from(self, source: "QuantizedSwitchGLUExecutor", expert_ids) -> None:
        self.copy_experts_into_slots(source, range(len(expert_ids)), expert_ids)

    def copy_experts_into_slots(self, source: "QuantizedSwitchGLUExecutor", slot_ids, expert_ids) -> None:
        for slot, expert_id in zip(slot_ids, expert_ids):
            self._gate_weight[slot] = source._gate_weight[expert_id]
            self._gate_scales_u16[slot] = source._gate_scales_u16[expert_id]
            self._gate_biases_u16[slot] = source._gate_biases_u16[expert_id]

            self._up_weight[slot] = source._up_weight[expert_id]
            self._up_scales_u16[slot] = source._up_scales_u16[expert_id]
            self._up_biases_u16[slot] = source._up_biases_u16[expert_id]

            self._down_weight[slot] = source._down_weight[expert_id]
            self._down_scales_u16[slot] = source._down_scales_u16[expert_id]
            self._down_biases_u16[slot] = source._down_biases_u16[expert_id]
        self._refresh_scale_bias_views()

    def resident_buffers(self) -> tuple[mx.array, ...]:
        return (
            self._gate_weight,
            self._gate_scales_u16,
            self._gate_biases_u16,
            self._up_weight,
            self._up_scales_u16,
            self._up_biases_u16,
            self._down_weight,
            self._down_scales_u16,
            self._down_biases_u16,
        )

    def _call_impl(self, x: mx.array, indices: mx.array, assume_sorted_indices: bool = False) -> mx.array:
        return quantized_switch_glu_bank_forward(
            x,
            indices,
            self.gate_proj.weight,
            self.gate_proj.scales,
            self.gate_proj.biases,
            self.up_proj.weight,
            self.up_proj.scales,
            self.up_proj.biases,
            self.down_proj.weight,
            self.down_proj.scales,
            self.down_proj.biases,
            group_size=self.gate_proj.group_size,
            gate_bits=self.gate_proj.bits,
            up_bits=self.up_proj.bits,
            down_bits=self.down_proj.bits,
            mode=self.gate_proj.mode,
            assume_sorted_indices=assume_sorted_indices,
        )

    def compiled(self, x: mx.array, indices: mx.array, assume_sorted_indices: bool = False) -> mx.array:
        cache_attr = "_compiled_sorted_call" if assume_sorted_indices else "_compiled_unsorted_call"
        compiled_fn = getattr(self, cache_attr)
        if compiled_fn is None:
            compiled_fn = mx.compile(
                lambda in_x, in_indices, gate_weight, gate_scales, gate_biases, up_weight, up_scales, up_biases, down_weight, down_scales, down_biases: quantized_switch_glu_bank_forward(
                    in_x,
                    in_indices,
                    gate_weight,
                    gate_scales,
                    gate_biases,
                    up_weight,
                    up_scales,
                    up_biases,
                    down_weight,
                    down_scales,
                    down_biases,
                    group_size=self.gate_proj.group_size,
                    gate_bits=self.gate_proj.bits,
                    up_bits=self.up_proj.bits,
                    down_bits=self.down_proj.bits,
                    mode=self.gate_proj.mode,
                    assume_sorted_indices=assume_sorted_indices,
                ),
            )
            setattr(self, cache_attr, compiled_fn)
        return compiled_fn(
            x,
            indices,
            self.gate_proj.weight,
            self.gate_proj.scales,
            self.gate_proj.biases,
            self.up_proj.weight,
            self.up_proj.scales,
            self.up_proj.biases,
            self.down_proj.weight,
            self.down_proj.scales,
            self.down_proj.biases,
        )

    def __call__(self, x: mx.array, indices: mx.array, assume_sorted_indices: bool = False) -> mx.array:
        return self._call_impl(x, indices, assume_sorted_indices=assume_sorted_indices)
