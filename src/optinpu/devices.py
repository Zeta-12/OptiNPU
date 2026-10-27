"""Device discovery for training (PyTorch) and inference (OpenVINO)."""

from __future__ import annotations

from functools import lru_cache

import openvino as ov


@lru_cache(maxsize=1)
def core() -> ov.Core:
    """Shared OpenVINO core (creating one is slow)."""
    return ov.Core()


def available_devices() -> dict[str, str]:
    """OpenVINO inference devices, e.g. {"CPU": "...", "GPU": "...", "NPU": "Intel(R) AI Boost"}."""
    c = core()
    return {d: c.get_property(d, "FULL_DEVICE_NAME") for d in c.available_devices}


def has_npu() -> bool:
    return "NPU" in core().available_devices


def torch_device(preferred: str | None = None):
    """Best PyTorch device for training: explicit choice, else xpu (Intel iGPU), cuda, cpu.

    The Intel NPU cannot run PyTorch training, so it is never returned here.
    """
    import torch

    if preferred:
        return torch.device(preferred)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_ov_device(device: str) -> str:
    """Accept "npu", "gpu", "cpu", "auto" in any case; return the OpenVINO name."""
    d = device.upper()
    if d == "AUTO":
        return "AUTO"
    if d.split(".")[0] not in {"CPU", "GPU", "NPU"}:
        raise ValueError(f"Unknown inference device {device!r}; expected cpu, gpu, npu or auto")
    return d
