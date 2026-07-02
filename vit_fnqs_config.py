from transformers import PretrainedConfig
from typing import List


class ViTFNQSConfig(PretrainedConfig):
    model_type = "vit_fnqs"

    def __init__(
        self,
        L_eff=25,
        num_layers = 4,
        d_model = 72,
        heads = 12,
        b = 2,
        complex: bool = True,
        disorder: bool = False,
        tras_inv = True,
        two_dim = True,
        **kwargs,
    ):
        self.L_eff = L_eff
        self.num_layers = num_layers
        self.d_model = d_model
        self.heads = heads
        self.b = b
        self.complex = complex
        self.disorder = disorder
        self.tras_inv = tras_inv
        self.two_dim = two_dim
        super().__init__(**kwargs)
