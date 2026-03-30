from __future__ import annotations

import ctypes
import json
import mmap
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Protocol

import numpy as np


_ROOT_DIR = Path(__file__).resolve().parent.parent
_SRC_PATH = _ROOT_DIR / "csrc" / "expert_io.c"
_BUILD_DIR = _ROOT_DIR / "build"
_DYLIB_PATH = _BUILD_DIR / "libflashmoe_expert_io.dylib"

if not _SRC_PATH.exists():
    _SRC_PATH = _ROOT_DIR / "expert_io.c"


@dataclass(frozen=True)
class ExpertGeometry:
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    group_size: int
    bits: int
    mode: str = "affine"

    @property
    def values_per_uint32(self) -> int:
        return 32 // self.bits

    @property
    def packed_hidden_size(self) -> int:
        return self.hidden_size // self.values_per_uint32

    @property
    def packed_moe_size(self) -> int:
        return self.moe_intermediate_size // self.values_per_uint32

    @property
    def hidden_groups(self) -> int:
        return self.hidden_size // self.group_size

    @property
    def moe_groups(self) -> int:
        return self.moe_intermediate_size // self.group_size

    @property
    def gate_weight_shape(self) -> tuple[int, int]:
        return (self.moe_intermediate_size, self.packed_hidden_size)

    @property
    def gate_scale_shape(self) -> tuple[int, int]:
        return (self.moe_intermediate_size, self.hidden_groups)

    @property
    def up_weight_shape(self) -> tuple[int, int]:
        return (self.moe_intermediate_size, self.packed_hidden_size)

    @property
    def up_scale_shape(self) -> tuple[int, int]:
        return (self.moe_intermediate_size, self.hidden_groups)

    @property
    def down_weight_shape(self) -> tuple[int, int]:
        return (self.hidden_size, self.packed_moe_size)

    @property
    def down_scale_shape(self) -> tuple[int, int]:
        return (self.hidden_size, self.moe_groups)

    @property
    def gate_weight_bytes(self) -> int:
        return int(np.prod(self.gate_weight_shape)) * np.dtype(np.uint32).itemsize

    @property
    def gate_scale_bytes(self) -> int:
        return int(np.prod(self.gate_scale_shape)) * np.dtype(np.uint16).itemsize

    @property
    def gate_bias_bytes(self) -> int:
        return self.gate_scale_bytes

    @property
    def up_weight_bytes(self) -> int:
        return int(np.prod(self.up_weight_shape)) * np.dtype(np.uint32).itemsize

    @property
    def up_scale_bytes(self) -> int:
        return int(np.prod(self.up_scale_shape)) * np.dtype(np.uint16).itemsize

    @property
    def up_bias_bytes(self) -> int:
        return self.up_scale_bytes

    @property
    def down_weight_bytes(self) -> int:
        return int(np.prod(self.down_weight_shape)) * np.dtype(np.uint32).itemsize

    @property
    def down_scale_bytes(self) -> int:
        return int(np.prod(self.down_scale_shape)) * np.dtype(np.uint16).itemsize

    @property
    def down_bias_bytes(self) -> int:
        return self.down_scale_bytes

    @property
    def gate_weight_offset(self) -> int:
        return 0

    @property
    def gate_scale_offset(self) -> int:
        return self.gate_weight_offset + self.gate_weight_bytes

    @property
    def gate_bias_offset(self) -> int:
        return self.gate_scale_offset + self.gate_scale_bytes

    @property
    def up_weight_offset(self) -> int:
        return self.gate_bias_offset + self.gate_bias_bytes

    @property
    def up_scale_offset(self) -> int:
        return self.up_weight_offset + self.up_weight_bytes

    @property
    def up_bias_offset(self) -> int:
        return self.up_scale_offset + self.up_scale_bytes

    @property
    def down_weight_offset(self) -> int:
        return self.up_bias_offset + self.up_bias_bytes

    @property
    def down_scale_offset(self) -> int:
        return self.down_weight_offset + self.down_weight_bytes

    @property
    def down_bias_offset(self) -> int:
        return self.down_scale_offset + self.down_scale_bytes

    @property
    def expert_size(self) -> int:
        return self.down_bias_offset + self.down_bias_bytes

    @property
    def layer_file_size(self) -> int:
        return self.expert_size * self.num_experts

    def validate(self) -> None:
        if self.mode != "affine":
            raise ValueError(f"Unsupported expert quantization mode: {self.mode}")
        if self.bits not in (2, 4):
            raise ValueError(f"Only 2-bit and 4-bit experts are supported, received {self.bits}")
        if self.hidden_size % self.group_size != 0:
            raise ValueError("hidden_size must be divisible by group_size")
        if self.moe_intermediate_size % self.group_size != 0:
            raise ValueError("moe_intermediate_size must be divisible by group_size")
        if self.hidden_size % self.values_per_uint32 != 0:
            raise ValueError("hidden_size must align with quantized packing")
        if self.moe_intermediate_size % self.values_per_uint32 != 0:
            raise ValueError("moe_intermediate_size must align with quantized packing")


@dataclass(frozen=True)
class ExpertSlotView:
    gate_weight: np.ndarray
    gate_scales_bf16: np.ndarray
    gate_biases_bf16: np.ndarray
    up_weight: np.ndarray
    up_scales_bf16: np.ndarray
    up_biases_bf16: np.ndarray
    down_weight: np.ndarray
    down_scales_bf16: np.ndarray
    down_biases_bf16: np.ndarray


@dataclass(frozen=True)
class SlotBankLoadResult:
    slot_ids: list[int]
    miss_slot_ids: list[int]
    miss_expert_ids: list[int]
    miss_views: list[ExpertSlotView]


class ExpertLoaderProtocol(Protocol):
    experts_dir: Path
    num_layers: int
    geometry: ExpertGeometry
    max_k: int

    def close(self) -> None: ...
    def load_layer(self, layer_index: int, expert_ids: list[int]) -> list[memoryview]: ...
    def load_layer_views(self, layer_index: int, expert_ids: list[int]) -> list[ExpertSlotView]: ...


def _build_native_helper() -> Path:
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    needs_rebuild = not _DYLIB_PATH.exists() or _SRC_PATH.stat().st_mtime > _DYLIB_PATH.stat().st_mtime
    if needs_rebuild:
        subprocess.run(
            [
                "clang",
                "-O3",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-fPIC",
                "-dynamiclib",
                str(_SRC_PATH),
                "-o",
                str(_DYLIB_PATH),
                "-lpthread",
            ],
            check=True,
        )
    return _DYLIB_PATH


def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_build_native_helper()))
    lib.flash_moe_expert_loader_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.flash_moe_expert_loader_create.restype = ctypes.c_void_p
    lib.flash_moe_expert_loader_destroy.argtypes = [ctypes.c_void_p]
    lib.flash_moe_expert_loader_destroy.restype = None
    lib.flash_moe_expert_loader_set_cache_io_split.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.flash_moe_expert_loader_set_cache_io_split.restype = ctypes.c_int
    lib.flash_moe_expert_loader_expert_size.argtypes = [ctypes.c_void_p]
    lib.flash_moe_expert_loader_expert_size.restype = ctypes.c_size_t
    lib.flash_moe_expert_loader_get_slot_buffer.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.flash_moe_expert_loader_get_slot_buffer.restype = ctypes.c_void_p
    lib.flash_moe_expert_loader_load.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.flash_moe_expert_loader_load.restype = ctypes.c_int
    lib.flash_moe_expert_loader_enable_slot_bank.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.flash_moe_expert_loader_enable_slot_bank.restype = ctypes.c_int
    lib.flash_moe_expert_loader_slot_bank_load.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.flash_moe_expert_loader_slot_bank_load.restype = ctypes.c_int
    return lib


class NativeExpertLoader:
    def __init__(
        self,
        experts_dir: Path,
        num_layers: int,
        geometry: ExpertGeometry,
        max_k: int,
        cache_io_split: int = 1,
    ) -> None:
        geometry.validate()
        self.experts_dir = Path(experts_dir)
        self.num_layers = num_layers
        self.geometry = geometry
        self.max_k = max_k
        self._lib = _load_library()
        self._handle = self._lib.flash_moe_expert_loader_create(
            str(self.experts_dir).encode("utf-8"),
            self.num_layers,
            self.geometry.expert_size,
            self.max_k,
            cache_io_split,
        )
        if not self._handle:
            raise RuntimeError(f"Failed to initialize native expert loader for {self.experts_dir}")

        self._slot_buffers: List[memoryview] = []
        self._slot_ctypes: List[ctypes.Array] = []
        buf_type = ctypes.c_ubyte * self.geometry.expert_size
        for slot in range(self.max_k):
            addr = self._lib.flash_moe_expert_loader_get_slot_buffer(self._handle, slot)
            if not addr:
                raise RuntimeError(f"Native expert loader returned null slot buffer for slot {slot}")
            c_buf = buf_type.from_address(addr)
            self._slot_ctypes.append(c_buf)
            self._slot_buffers.append(memoryview(c_buf))
        self._slot_views: List[ExpertSlotView] = [
            unpack_expert_slot(slot_buffer, self.geometry) for slot_buffer in self._slot_buffers
        ]

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.flash_moe_expert_loader_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def set_cache_io_split(self, cache_io_split: int) -> None:
        rc = self._lib.flash_moe_expert_loader_set_cache_io_split(self._handle, cache_io_split)
        if rc != 0:
            raise RuntimeError(f"Failed to set cache_io_split={cache_io_split}")

    def load_layer(self, layer_index: int, expert_ids: list[int]) -> list[memoryview]:
        if not expert_ids:
            return []
        if len(expert_ids) > self.max_k:
            raise ValueError(f"Requested {len(expert_ids)} experts but loader max_k is {self.max_k}")

        expert_arr = (ctypes.c_int * len(expert_ids))(*expert_ids)
        valid_arr = (ctypes.c_int * len(expert_ids))()
        loaded = self._lib.flash_moe_expert_loader_load(
            self._handle,
            layer_index,
            expert_arr,
            len(expert_ids),
            valid_arr,
        )
        if loaded != len(expert_ids):
            valid = [int(v) for v in valid_arr]
            raise RuntimeError(
                f"Failed to load all routed experts for layer {layer_index}: "
                f"requested={len(expert_ids)} loaded={loaded} valid={valid}"
            )
        return self._slot_buffers[: len(expert_ids)]

    def slot_view(self, slot_index: int) -> memoryview:
        return self._slot_buffers[slot_index]

    def slot_expert_view(self, slot_index: int) -> ExpertSlotView:
        return self._slot_views[slot_index]

    def load_layer_views(self, layer_index: int, expert_ids: list[int]) -> list[ExpertSlotView]:
        self.load_layer(layer_index, expert_ids)
        return self._slot_views[: len(expert_ids)]


class NativeSlotBankLoader(NativeExpertLoader):
    def __init__(
        self,
        experts_dir: Path,
        num_layers: int,
        geometry: ExpertGeometry,
        slot_bank_size: int,
        cache_io_split: int = 1,
    ) -> None:
        super().__init__(
            experts_dir=experts_dir,
            num_layers=num_layers,
            geometry=geometry,
            max_k=slot_bank_size,
            cache_io_split=cache_io_split,
        )
        self.slot_bank_size = slot_bank_size
        rc = self._lib.flash_moe_expert_loader_enable_slot_bank(self._handle, self.slot_bank_size)
        if rc != 0:
            self.close()
            raise RuntimeError(
                f"Native expert loader failed to enable slot bank of size {self.slot_bank_size}"
            )

    def slot_bank_load(self, layer_index: int, expert_ids: list[int]) -> SlotBankLoadResult:
        if layer_index < 0 or layer_index >= self.num_layers:
            raise ValueError(f"layer_index out of range: {layer_index}")
        if not expert_ids:
            raise ValueError("slot_bank_load requires at least one expert id")
        if len(expert_ids) > self.slot_bank_size:
            raise ValueError(
                f"Requested {len(expert_ids)} experts, but slot bank size is {self.slot_bank_size}"
            )

        expert_idx_array = (ctypes.c_int * len(expert_ids))(*expert_ids)
        slot_array = (ctypes.c_int * len(expert_ids))()
        miss_array = (ctypes.c_int * len(expert_ids))()
        rc = self._lib.flash_moe_expert_loader_slot_bank_load(
            self._handle,
            layer_index,
            expert_idx_array,
            len(expert_ids),
            slot_array,
            miss_array,
        )
        if rc < 0:
            raise RuntimeError(
                f"Native slot-bank load failed for layer {layer_index} experts {expert_ids}"
            )

        slot_ids = [int(slot_array[i]) for i in range(len(expert_ids))]
        miss_slot_ids: list[int] = []
        miss_expert_ids: list[int] = []
        miss_views: list[ExpertSlotView] = []
        for pos, expert_id in enumerate(expert_ids):
            if not miss_array[pos]:
                continue
            slot_id = slot_ids[pos]
            miss_slot_ids.append(slot_id)
            miss_expert_ids.append(expert_id)
            miss_views.append(self.slot_expert_view(slot_id))
        return SlotBankLoadResult(
            slot_ids=slot_ids,
            miss_slot_ids=miss_slot_ids,
            miss_expert_ids=miss_expert_ids,
            miss_views=miss_views,
        )


class ResidentFlashExpertLoader:
    def __init__(
        self,
        experts_dir: Path,
        num_layers: int,
        geometry: ExpertGeometry,
        max_k: int,
    ) -> None:
        geometry.validate()
        self.experts_dir = Path(experts_dir)
        self.num_layers = num_layers
        self.geometry = geometry
        self.max_k = max_k
        self.layer_paths = [self.experts_dir / f"layer_{layer:02d}.bin" for layer in range(num_layers)]
        self._layer_bytes = [path.read_bytes() for path in self.layer_paths]
        self._layer_views = [memoryview(layer_bytes) for layer_bytes in self._layer_bytes]
        self.total_bytes = sum(len(layer_bytes) for layer_bytes in self._layer_bytes)

    def close(self) -> None:
        self._layer_views = []
        self._layer_bytes = []

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def load_layer(self, layer_index: int, expert_ids: list[int]) -> list[memoryview]:
        if not expert_ids:
            return []
        if len(expert_ids) > self.max_k:
            raise ValueError(f"Requested {len(expert_ids)} experts but loader max_k is {self.max_k}")
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Layer index out of range: {layer_index}")

        layer_view = self._layer_views[layer_index]
        expert_size = self.geometry.expert_size
        slots: list[memoryview] = []
        for expert_id in expert_ids:
            if expert_id < 0 or expert_id >= self.geometry.num_experts:
                raise IndexError(f"Expert id out of range: {expert_id}")
            start = expert_id * expert_size
            end = start + expert_size
            slots.append(layer_view[start:end])
        return slots

    def load_layer_views(self, layer_index: int, expert_ids: list[int]) -> list[ExpertSlotView]:
        return [unpack_expert_slot(slot_buffer, self.geometry) for slot_buffer in self.load_layer(layer_index, expert_ids)]


@dataclass(frozen=True)
class _MixedSidecarTensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    bits: int

    @property
    def per_expert_nbytes(self) -> int:
        if not self.shape:
            return self.nbytes
        experts = self.shape[0]
        if experts <= 0 or self.nbytes % experts != 0:
            raise ValueError(f"Invalid per-expert size for {self.name}")
        return self.nbytes // experts


class MixedPrecisionSidecarLoader:
    def __init__(self, experts_dir: Path, num_layers: int, max_k: int) -> None:
        self.experts_dir = Path(experts_dir)
        self.num_layers = num_layers
        self.max_k = max_k
        layout_path = self.experts_dir / "layout.json"
        if not layout_path.exists():
            raise FileNotFoundError(f"Missing mixed sidecar layout: {layout_path}")
        with layout_path.open("r", encoding="utf-8") as f:
            self.layout = json.load(f)
        self.format = str(self.layout.get("format", ""))
        if self.format != "mlx-flash-moe-mixed-sidecar-v1":
            raise ValueError(f"Unsupported mixed sidecar format: {self.format}")
        if int(self.layout["num_layers"]) != self.num_layers:
            raise ValueError(
                f"Mixed sidecar layer count {self.layout['num_layers']} does not match model {self.num_layers}"
            )
        self.num_experts = int(self.layout["num_experts"])
        self.group_size = int(self.layout.get("group_size", 64))
        self._layer_files: list[Any] = []
        self._layer_mm: list[mmap.mmap] = []
        self._layers: list[dict[str, _MixedSidecarTensorRecord]] = []
        self._layer_bits: list[tuple[int, int, int]] = []

        self._open_layers()

    def _open_layers(self) -> None:
        self._layer_files = []
        self._layer_mm = []
        self._layers = []
        self._layer_bits = []
        for layer in self.layout["layers"]:
            path = self.experts_dir / layer["file"]
            f = path.open("rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._layer_files.append(f)
            self._layer_mm.append(mm)
            record_map: dict[str, _MixedSidecarTensorRecord] = {}
            gate_bits = up_bits = down_bits = 0
            for tensor in layer["tensors"]:
                name = str(tensor["name"])
                record = _MixedSidecarTensorRecord(
                    name=name,
                    dtype=str(tensor["dtype"]),
                    shape=tuple(int(dim) for dim in tensor["shape"]),
                    offset=int(tensor["offset"]),
                    nbytes=int(tensor["nbytes"]),
                    bits=int(tensor.get("bits", 0)),
                )
                record_map[name] = record
                if name.endswith(".weight"):
                    proj = name.split(".")[-2]
                    if proj == "gate_proj":
                        gate_bits = record.bits
                    elif proj == "up_proj":
                        up_bits = record.bits
                    elif proj == "down_proj":
                        down_bits = record.bits
            self._layers.append(record_map)
            self._layer_bits.append((gate_bits, up_bits, down_bits))

    def close(self) -> None:
        for mm in self._layer_mm:
            try:
                mm.close()
            except Exception:
                pass
        for f in self._layer_files:
            try:
                f.close()
            except Exception:
                pass
        self._layer_mm = []
        self._layer_files = []

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def layer_bits(self, layer_index: int) -> tuple[int, int, int]:
        return self._layer_bits[layer_index]

    def _tensor_view(self, layer_index: int, tensor_name: str, expert_id: int) -> np.ndarray:
        record = self._layers[layer_index][tensor_name]
        if expert_id < 0 or expert_id >= self.num_experts:
            raise IndexError(f"Expert id out of range: {expert_id}")
        tensor_shape = record.shape
        if not tensor_shape:
            raise ValueError(f"Invalid tensor shape for {tensor_name}")
        slice_shape = tensor_shape[1:]
        slice_nbytes = record.per_expert_nbytes
        offset = record.offset + expert_id * slice_nbytes
        dtype = np.uint16 if record.dtype == "bfloat16" else np.dtype(record.dtype)
        return np.frombuffer(
            self._layer_mm[layer_index],
            dtype=dtype,
            count=int(np.prod(slice_shape)),
            offset=offset,
        ).reshape(slice_shape)

    def load_layer_views(self, layer_index: int, expert_ids: list[int]) -> list[ExpertSlotView]:
        if not expert_ids:
            return []
        if len(expert_ids) > self.max_k:
            raise ValueError(f"Requested {len(expert_ids)} experts but loader max_k is {self.max_k}")
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(f"Layer index out of range: {layer_index}")

        views: list[ExpertSlotView] = []
        for expert_id in expert_ids:
            views.append(
                ExpertSlotView(
                    gate_weight=self._tensor_view(
                        layer_index, f"language_model.model.layers.{layer_index}.mlp.switch_mlp.gate_proj.weight", expert_id
                    ),
                    gate_scales_bf16=self._tensor_view(
                        layer_index, f"language_model.model.layers.{layer_index}.mlp.switch_mlp.gate_proj.scales", expert_id
                    ),
                    gate_biases_bf16=self._tensor_view(
                        layer_index, f"language_model.model.layers.{layer_index}.mlp.switch_mlp.gate_proj.biases", expert_id
                    ),
                    up_weight=self._tensor_view(
                        layer_index, f"language_model.model.layers.{layer_index}.mlp.switch_mlp.up_proj.weight", expert_id
                    ),
                    up_scales_bf16=self._tensor_view(
                        layer_index, f"language_model.model.layers.{layer_index}.mlp.switch_mlp.up_proj.scales", expert_id
                    ),
                    up_biases_bf16=self._tensor_view(
                        layer_index, f"language_model.model.layers.{layer_index}.mlp.switch_mlp.up_proj.biases", expert_id
                    ),
                    down_weight=self._tensor_view(
                        layer_index, f"language_model.model.layers.{layer_index}.mlp.switch_mlp.down_proj.weight", expert_id
                    ),
                    down_scales_bf16=self._tensor_view(
                        layer_index, f"language_model.model.layers.{layer_index}.mlp.switch_mlp.down_proj.scales", expert_id
                    ),
                    down_biases_bf16=self._tensor_view(
                        layer_index, f"language_model.model.layers.{layer_index}.mlp.switch_mlp.down_proj.biases", expert_id
                    ),
                )
            )
        return views

    def load_layer(self, layer_index: int, expert_ids: list[int]) -> list[memoryview]:
        raise NotImplementedError(
            "MixedPrecisionSidecarLoader exposes per-expert views directly; use load_layer_views()"
        )


def unpack_expert_slot(slot_buffer: memoryview, geometry: ExpertGeometry) -> ExpertSlotView:
    return ExpertSlotView(
        gate_weight=np.frombuffer(
            slot_buffer,
            dtype=np.uint32,
            count=int(np.prod(geometry.gate_weight_shape)),
            offset=geometry.gate_weight_offset,
        ).reshape(geometry.gate_weight_shape),
        gate_scales_bf16=np.frombuffer(
            slot_buffer,
            dtype=np.uint16,
            count=int(np.prod(geometry.gate_scale_shape)),
            offset=geometry.gate_scale_offset,
        ).reshape(geometry.gate_scale_shape),
        gate_biases_bf16=np.frombuffer(
            slot_buffer,
            dtype=np.uint16,
            count=int(np.prod(geometry.gate_scale_shape)),
            offset=geometry.gate_bias_offset,
        ).reshape(geometry.gate_scale_shape),
        up_weight=np.frombuffer(
            slot_buffer,
            dtype=np.uint32,
            count=int(np.prod(geometry.up_weight_shape)),
            offset=geometry.up_weight_offset,
        ).reshape(geometry.up_weight_shape),
        up_scales_bf16=np.frombuffer(
            slot_buffer,
            dtype=np.uint16,
            count=int(np.prod(geometry.up_scale_shape)),
            offset=geometry.up_scale_offset,
        ).reshape(geometry.up_scale_shape),
        up_biases_bf16=np.frombuffer(
            slot_buffer,
            dtype=np.uint16,
            count=int(np.prod(geometry.up_scale_shape)),
            offset=geometry.up_bias_offset,
        ).reshape(geometry.up_scale_shape),
        down_weight=np.frombuffer(
            slot_buffer,
            dtype=np.uint32,
            count=int(np.prod(geometry.down_weight_shape)),
            offset=geometry.down_weight_offset,
        ).reshape(geometry.down_weight_shape),
        down_scales_bf16=np.frombuffer(
            slot_buffer,
            dtype=np.uint16,
            count=int(np.prod(geometry.down_scale_shape)),
            offset=geometry.down_scale_offset,
        ).reshape(geometry.down_scale_shape),
        down_biases_bf16=np.frombuffer(
            slot_buffer,
            dtype=np.uint16,
            count=int(np.prod(geometry.down_scale_shape)),
            offset=geometry.down_bias_offset,
        ).reshape(geometry.down_scale_shape),
    )
