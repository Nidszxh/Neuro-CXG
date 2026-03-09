__all__ = ["PipelineHealthCheck", "PipelineValidator", "ensure_atlas"]


def __getattr__(name: str):
    """Lazy-load heavy submodules to avoid the runpy sys.modules warning.

    When pipeline_checks (or atlas_validator) is executed via
    ``python -m src.validation.pipeline_checks``, Python imports the parent
    package ``src.validation`` first.  If that package eagerly imports the
    submodule, ``runpy`` finds it already in ``sys.modules`` and emits:

        RuntimeWarning: 'src.validation.pipeline_checks' found in sys.modules
        after import of package 'src.validation'

    Deferring all submodule imports to first-attribute-access avoids this.
    """
    if name in ("PipelineHealthCheck", "PipelineValidator"):
        from .pipeline_checks import (  # noqa: PLC0415  (intentional lazy import)
            PipelineHealthCheck as _PHC,
            PipelineValidator as _PV,
        )
        import sys as _sys
        _mod = _sys.modules[__name__]
        _mod.PipelineHealthCheck = _PHC  # type: ignore[union-attr]
        _mod.PipelineValidator = _PV     # type: ignore[union-attr]
        return _mod.__dict__[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ensure_atlas(*args, **kwargs):
	"""Lazy proxy to avoid preloading atlas_validator during package import.

	Importing ``src.validation`` should not import ``src.validation.atlas_validator``
	eagerly, because running the latter via ``python -m`` can otherwise trigger the
	runpy warning about preloaded modules in ``sys.modules``.
	"""
	from .atlas_validator import ensure_atlas as _ensure_atlas

	return _ensure_atlas(*args, **kwargs)


__all__ = ["PipelineHealthCheck", "PipelineValidator", "ensure_atlas"]
