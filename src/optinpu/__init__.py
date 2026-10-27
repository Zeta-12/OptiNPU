"""optinpu: train object detectors anywhere, run them on the Intel NPU."""

import os

# Intel GPU (XPU) kernels are JIT-compiled on first use (~17 min for D-FINE on an Arrow Lake iGPU).
# Without a persistent cache every new process pays that again. Must be set before torch loads.
os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")

from optinpu.devices import available_devices, torch_device
from optinpu.export import export_openvino
from optinpu.runtime import CompiledModel, compile_model
from optinpu.bench import benchmark
from optinpu.detector import Detector
from optinpu.models import list_models
from optinpu.postprocess import Detections

__version__ = "0.1.0"

__all__ = [
    "Detector",
    "Detections",
    "list_models",
    "available_devices",
    "torch_device",
    "export_openvino",
    "CompiledModel",
    "compile_model",
    "benchmark",
]
