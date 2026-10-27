"""OpenVINO inference runtime with compilation caching, single-device or hybrid."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import openvino as ov

from optinpu.devices import core, normalize_ov_device

DEFAULT_CACHE = Path(os.environ.get("OPTINPU_CACHE", Path.home() / ".cache" / "optinpu" / "ov"))


class CompiledModel:
    """A model compiled for one device. Call it with numpy arrays, get numpy arrays back."""

    def __init__(self, compiled: ov.CompiledModel, device: str):
        self.compiled = compiled
        self.device = device
        self._request = compiled.create_infer_request()

    @property
    def input_shapes(self) -> list[tuple[int, ...]]:
        return [tuple(i.get_shape()) for i in self.compiled.inputs]

    @property
    def output_names(self) -> list[str]:
        return [o.get_any_name() if o.get_names() else f"output{i}" for i, o in enumerate(self.compiled.outputs)]

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        return [v.copy() for v in self.views(*inputs)]

    def views(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Like __call__ without copies: outputs are only valid until the next call."""
        result = self._request.infer(list(inputs), share_inputs=True, share_outputs=True)
        return [result[o] for o in self.compiled.outputs]


class HybridModel:
    """Head compiled on one device (e.g. NPU), tail on another (e.g. GPU), run back to back."""

    def __init__(self, head: CompiledModel, tail: CompiledModel):
        self.head, self.tail = head, tail
        self.device = f"{head.device}+{tail.device}"

    @property
    def input_shapes(self) -> list[tuple[int, ...]]:
        return self.head.input_shapes

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        # head outputs are several MB: hand them to the tail without copying
        return self.tail(*self.head.views(*inputs))


def compile_model(
    model: str | Path | ov.Model,
    device: str = "npu",
    hint: str = "LATENCY",
    cache_dir: str | Path | None = DEFAULT_CACHE,
    config: dict | None = None,
) -> CompiledModel:
    """Compile an IR path or ov.Model for `device` ("cpu", "gpu", "npu", "auto").

    `hint` is the OpenVINO performance hint: "LATENCY" (batch 1, one stream) or "THROUGHPUT".
    `config` adds device properties (e.g. {"NPU_TILES": 1}).
    The compiled blob is cached on disk, which turns NPU compile time from seconds to milliseconds.
    """
    c = core()
    ov_device = normalize_ov_device(device)
    if isinstance(model, (str, Path)):
        model = c.read_model(model)

    full_config = {"PERFORMANCE_HINT": hint, **(config or {})}
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        full_config["CACHE_DIR"] = str(cache_dir)
    return CompiledModel(c.compile_model(model, ov_device, full_config), ov_device)


def compile_hybrid(
    model: str | Path | ov.Model,
    split: str,
    head_device: str = "npu",
    tail_device: str = "gpu",
    head_config: dict | None = None,
    tail_config: dict | None = None,
) -> HybridModel:
    """Split after the last op matching the regex `split`; compile each part on its device."""
    from optinpu.graph import split_after

    if isinstance(model, (str, Path)):
        model = core().read_model(model)
    head, tail = split_after(model, split)
    return HybridModel(
        compile_model(head, head_device, config=head_config),
        compile_model(tail, tail_device, config=tail_config),
    )
