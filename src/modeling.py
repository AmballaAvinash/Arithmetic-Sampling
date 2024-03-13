from transformers import GenerationMixin

from utils import ArithmeticSamplingGenerationMixin


def ArithmeticSampler(model_class: type) -> type:
    """
    Utility function for converting a model class into a class that inherits from `~generation.ArithmeticSamplingGenerationMixin`.
    """
    if not issubclass(model_class, GenerationMixin):
        raise ValueError(
            f"ArithmeticSampler() can only be applied to classes that inherit from `transformers.GenerationMixin`, "
            f"but got {model_class}."
        )
    return type("ArithmeticSampler" + model_class.__name__, (ArithmeticSamplingGenerationMixin, model_class), {})