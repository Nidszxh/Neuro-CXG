"""Validation package exports.

Exports are lazy-loaded to avoid preloading heavy submodules when running
``python -m src.validation.<module>`` entrypoints.
"""

__all__ = ["PipelineValidator"]


def __getattr__(name: str):
    """Lazy-load PipelineValidator on first attribute access."""
    if name == "PipelineValidator":
        import sys as _sys

        from .pipeline_checks import PipelineValidator as _PV  # noqa: PLC0415

        _mod = _sys.modules[__name__]
        _mod.PipelineValidator = _PV  # type: ignore[union-attr]
        return _mod.__dict__[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
