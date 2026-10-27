"""Model registry: name -> builder returning an export-ready torch module and its preprocessing spec."""

from optinpu.models.registry import ModelSpec, get_model, list_models

__all__ = ["ModelSpec", "get_model", "list_models"]
