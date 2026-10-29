"""Post-training INT8 quantization with NNCF."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import openvino as ov

from optinpu.devices import core


def quantize_model(
    model: str | Path | ov.Model,
    calibration: Iterable,
    transform: Callable[[object], np.ndarray],
    subset_size: int = 300,
    transformer: bool = True,
    ignored_types: Iterable[str] = (),
    ignored_patterns: Iterable[str] = (),
    ignored_names: Iterable[str] = (),
    **nncf_options,
) -> ov.Model:
    """INT8 version of an FP32/FP16 model (activations INT8, weights INT8), NNCF post-training.

    `calibration` items (e.g. image paths) are turned into model inputs by `transform`.
    `transformer=True` uses NNCF's transformer scheme (keeps LayerNorm/Softmax inputs in float).
    `ignored_types` / `ignored_patterns` / `ignored_names` keep matching ops (OpenVINO op types, name
    regexes, exact names) in float: mixed precision.
    `nncf_options` go to nncf.quantize (e.g. preset=nncf.QuantizationPreset.MIXED, fast_bias_correction=False).
    """
    import nncf

    if not isinstance(model, ov.Model):
        model = core().read_model(model)
    ignored_types, ignored_patterns, ignored_names = list(ignored_types), list(ignored_patterns), list(ignored_names)
    ignored = None
    if ignored_types or ignored_patterns or ignored_names:
        ignored = nncf.IgnoredScope(types=ignored_types, patterns=ignored_patterns, names=ignored_names,
                                    validate=False)
    calibration = list(calibration)
    return nncf.quantize(
        model,
        nncf.Dataset(calibration, transform),
        subset_size=min(subset_size, len(calibration)),
        model_type=nncf.ModelType.TRANSFORMER if transformer else None,
        ignored_scope=ignored,
        **nncf_options,
    )


def quantize_int8(
    ir: str | Path | ov.Model,
    calibration: Iterable,
    transform: Callable[[object], np.ndarray],
    output: str | Path,
    subset_size: int = 300,
    transformer: bool = True,
    ignored_types: Iterable[str] = (),
    ignored_patterns: Iterable[str] = (),
    **nncf_options,
) -> Path:
    """quantize_model, saved as an IR at `output` (weights compressed to FP16 where still float)."""
    quantized = quantize_model(ir, calibration, transform, subset_size, transformer, ignored_types, ignored_patterns,
                               **nncf_options)
    output = Path(output).with_suffix(".xml")
    output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, output, compress_to_fp16=True)
    return output
