import json

import torch
from transformers import PretrainedConfig
import matplotlib.pyplot as plt

from Binarized_MNIST_VAE.modeling_vae import VAEModel

config=PretrainedConfig(data_dim=784,
    latent_dim=20  ,         # VAE latent dimension
hidden_dim=1024,          # VAE Hidden units
encoder_layers=2,        # VAE Ecoder/De-coder network layers
data_type='auto' ,       # VAE choice of decoder - Bernauli for 'binary' and gaussian for 'continuous', 'auto' estiomate from data
epochs=44,)
model = VAEModel(config)
model_path = r"D:\git\fast-api-project\src\VAE_03\Binarized_MNIST_VAE\customVAE_model2.pth"
checkpoint = torch.load(f=model_path, map_location="cpu")
checkpoint_keys = checkpoint.keys()
print(list(checkpoint_keys))
print(checkpoint['training_info']['epochs'])
train_losses = checkpoint['training_info']['train_losses']
recon_losses = checkpoint['training_info']['recon_losses']
kl_losses = checkpoint['training_info']['kl_losses']

plt.plot(train_losses)
plt.plot(recon_losses)
plt.plot(kl_losses)
plt.show()
plt.close()

# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
#
# with torch.no_grad():
#     recon_x = model.generate()
#     print(recon_x.shape)
#     recon_x = recon_x.cpu().numpy().reshape(28, 28)
#     plt.imshow(recon_x, cmap='gray')
#     plt.show()

