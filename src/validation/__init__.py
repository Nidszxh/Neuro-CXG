"""Validation package exports.

Exports are lazy-loaded to avoid preloading heavy submodules when running
``python -m src.validation.<module>`` entrypoints.
"""

__all__ = ["PipelineValidator", "ensure_atlas"]


def __getattr__(name: str):
    """Lazy-load PipelineValidator on first attribute access."""
    if name == "PipelineValidator":
        from .pipeline_checks import PipelineValidator as _PV  # noqa: PLC0415
        import sys as _sys
        _mod = _sys.modules[__name__]
        _mod.PipelineValidator = _PV  # type: ignore[union-attr]
        return _mod.__dict__[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ensure_atlas(*args, **kwargs):
    """Lazy proxy for atlas validator entrypoint."""
    from .atlas_validator import ensure_atlas as _ensure_atlas

    return _ensure_atlas(*args, **kwargs)
