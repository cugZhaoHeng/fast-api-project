from transformers import PretrainedConfig
from typing import Optional, Literal

class VAEConfig(PretrainedConfig):
    model_type = "vae"
    
    def __init__(
        self,
        data_dim=784,
        latent_dim=20,
        hidden_dim=1024,
        encoder_layers=2,
        data_type: Optional[Literal['binary', 'continuous', 'auto']] = 'auto',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.data_type = data_type