from transformers.generation import GenerationConfig

class ArithmeticSamplingGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arithmetic_sampling = kwargs.pop("arithmetic_sampling", False)