"""Model registry: name -> builder returning an export-ready torch module and its preprocessing spec."""

__all__ = ["ModelSpec", "get_model", "list_models"]


def __getattr__(name):
    # lazy: the registry imports PyTorch (~8 s), which inference-only code (optinpu.tasks) does not need
    if name in __all__:
        from optinpu.models import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
