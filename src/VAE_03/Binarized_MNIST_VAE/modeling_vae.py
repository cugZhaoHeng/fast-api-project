# modeling_vae.py
from transformers import PreTrainedModel
import torch.nn as nn
import torch
import json
import os
from typing import Tuple, Optional, Literal
from .configuration_vae import VAEConfig
from huggingface_hub import hf_hub_download


class VAEModel(PreTrainedModel):
    config_class = VAEConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        self.latent_dim = config.latent_dim
        self.encoder_layers = config.encoder_layers
        self.data_type = config.data_type
        self.data_dim = config.data_dim
        self.hidden_dim=config.hidden_dim

        # Encoder
        currentDim = self.data_dim
        layers = []
        for i in range(self.encoder_layers):
            nextDim = self.hidden_dim if i ==0 else self.hidden_dim//2
            layers.append(nn.Linear(currentDim, nextDim))
            layers.append(nn.Tanh())
            currentDim = nextDim
        self.encodeLayers=nn.Sequential(*layers)
        self.fc_mu = nn.Linear(currentDim, self.latent_dim)
        self.fc_logvar = nn.Linear(currentDim, self.latent_dim)

        # Decoder for binary data
        currentDim = self.latent_dim
        layers = []
        for i in range(self.encoder_layers-1):
            nextDim = self.hidden_dim
            layers.append(nn.Linear(currentDim, nextDim))
            layers.append(nn.Tanh())
            currentDim = nextDim
        layers.append(nn.Linear(self.hidden_dim, self.data_dim))
        layers.append(nn.Sigmoid())
        self.decoder_bernoulli = nn.Sequential(*layers)

        # Decoder for continuous data
        currentDim = self.latent_dim
        layers = []
        for i in range(self.encoder_layers):
            nextDim = self.hidden_dim
            layers.append(nn.Linear(currentDim, nextDim))
            layers.append(nn.Tanh())
            currentDim = nextDim
        self.decoder_gaussian_layers = nn.Sequential(*layers)
        self.decoder_gaussian_mean = nn.Linear(self.hidden_dim, self.data_dim)
        self.decoder_gaussian_logvar = nn.Linear(self.hidden_dim, self.data_dim)

        self.prior_mean = torch.zeros(self.latent_dim)
        self.prior_std = torch.ones(self.latent_dim)

    def detect_data_type(self, x: torch.Tensor) -> str:
        unique_vals =torch.unique(x[0:2].flatten())
        if len(unique_vals) <= 2:
            print(f"Auto-detected: Binary data (unique values: {unique_vals.tolist()})")
            return 'binary'
        else:
            print(f"Auto-detected: Continuous data ({len(unique_vals)} unique values)")
            return 'continuous'

    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encodeLayers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if (self.data_type is None)or(self.data_type=='auto') :
            self.data_type = self.detect_data_type(z)
        if self.data_type == 'binary':
            return self.decoder_bernoulli(z), None
        else:
            h = self.decoder_gaussian_layers(z)
            return self.decoder_gaussian_mean(h), self.decoder_gaussian_logvar(h)
    
    def sample_prior(self, num_samples: int) -> torch.Tensor:
        return torch.randn(num_samples, self.latent_dim)

    def forward(self,x: torch.Tensor,data_type: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (self.data_type is None)or(self.data_type=='auto') :
            self.data_type = self.detect_data_type(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def reconstruction_loss(self, x: torch.Tensor, recon_output, mu: torch.Tensor, 
                          logvar: torch.Tensor, data_type: Optional[str] = None) -> torch.Tensor:
        if data_type is None:
            data_type = self.data_type
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if data_type == 'binary':
            if isinstance(recon_output, tuple):
                recon_output=recon_output[0]
            recon_loss = nn.functional.binary_cross_entropy(recon_output, x, reduction='sum')
        else:  # 'continuous'
            mean, logvar_x = recon_output
            var_x = torch.exp(logvar_x)
            recon_loss = 0.5 * torch.sum(torch.log(2 * torch.pi * var_x) + (x - mean).pow(2) / var_x)
        return recon_loss + kl_loss,recon_loss,kl_loss
    
    def generate(self, num_samples: int = 1, z: Optional[torch.Tensor] = None):
        if z is None:
            z = self.sample_prior(num_samples)
        recon_x = self.decode(z)
        if isinstance(recon_x, tuple):
            return recon_x[0]
        return recon_x
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Custom loading to handle your model format
        if os.path.isdir(pretrained_model_name_or_path):
            # Local directory
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            model_path = os.path.join(pretrained_model_name_or_path, "customVAE_model2.pth")
        else:
            # Hugging Face Hub model ID
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json",
                cache_dir=kwargs.get("cache_dir", None)
            )
            model_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="customVAE_model2.pth",
                cache_dir=kwargs.get("cache_dir", None)
            )
        
        # Load config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config
        config = VAEConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(model_path)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        model.load_state_dict(state_dict)
        return model