"""Public inference API, with compatibility for the original checkpoint families."""

from .cont_infer import ContInfer as PraxisPolicy

__all__ = ["PraxisPolicy"]
