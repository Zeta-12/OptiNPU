"""optinpu: train object detectors anywhere, run them on the Intel NPU."""

import os

# Intel GPU (XPU) kernels are JIT-compiled on first use (~17 min for D-FINE on an Arrow Lake iGPU).
# Without a persistent cache every new process pays that again. Must be set before torch loads.
os.environ.setdefault("SYCL_CACHE_PERSISTENT", "1")

from optinpu.devices import available_devices, torch_device  # noqa: E402
from optinpu.export import export_openvino  # noqa: E402
from optinpu.runtime import CompiledModel, HybridModel, compile_hybrid, compile_model  # noqa: E402
from optinpu.bench import benchmark  # noqa: E402
from optinpu.lint import lint  # noqa: E402
from optinpu.postprocess import Detections  # noqa: E402
from optinpu.profiler import profile_measured  # noqa: E402
from optinpu.optimizer import optimize  # noqa: E402

__version__ = "0.2.0"

__all__ = [
    "Detector",
    "Detections",
    "list_models",
    "available_devices",
    "torch_device",
    "export_openvino",
    "CompiledModel",
    "HybridModel",
    "compile_model",
    "compile_hybrid",
    "benchmark",
    "lint",
    "profile_measured",
    "optimize",
]


def __getattr__(name):
    # Detector and the model registry import PyTorch (~8 s): only on demand, so that `import optinpu`
    # stays cheap for inference and for the sandboxed jobs, each of which starts a new process
    if name == "Detector":
        from optinpu.detector import Detector

        return Detector
    if name == "list_models":
        from optinpu.models import list_models

        return list_models
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
