---
language: en
tags:
- vae
- generative-model
- pytorch
- mnist
- unsupervised-learning
license: mit
datasets:
- mnist
---

# VAE Model for MNIST

This is a Variational Autoencoder (VAE) model trained on the MNIST dataset.

## Model Description

This repository contains a complete implementation of a Variational Autoencoder (VAE) trained on the MNIST handwritten digits dataset. The model learns to encode images into a 2-dimensional latent space and decode them back to reconstructed images, enabling both data compression and generation of new digit-like images.
The architecture is based on the implementation outlined in **Auto-Encoding Variational Bayes by Diederik et al., 2022**

### Architecture Details

- **Model Type**: Variational Autoencoder (VAE)
- **Framework**: PyTorch
- **Input**: 28×28 grayscale images (784 dimensions)
- **Latent Space**: 20 dimensions 
- **Encoder and Decoder Layers**: 2 
- **Encoder and Decoder Hidden Units**: 1024 → 512 (encoder), 1024 → 512 (decoder)
- **Total Parameters**: ~4.8M
- **Data type:** Binary/Continous (automatically detected)
- **Current Implementation:** Binary (pixel>0.5)

### Key Components

1. **Encoder Network**: Maps input images to latent distribution parameters (μ, σ²)
2. **Reparameterization Trick**: Enables differentiable sampling from the latent distribution
3. **Decoder Network**: Reconstructs images from latent space samples
4. **Loss Function**: Combines reconstruction loss ELBO (Bernoulli: binary cross-entropy,  Gaussian: negative log-likelihood) + KL divergence

## Training Details

- **Dataset**: MNIST (60,000 training images, 10,000 test images) torchvision.datasets.MNIST
- **Batch Size**: 128
- **Epochs**: 44
- **Optimizer**: Adam
- **Learning Rate**: 1e-3

## Model Performance

### Metrics
- **Final Training Loss**: ~79.6
- **Final Validation Loss**: ~84.3
- **Reconstruction Loss**: ~48.0
- **KL Divergence**: ~31.5

### Capabilities
- ✅ High-quality digit reconstruction
- ✅ Smooth latent space interpolation
- ✅ Generation of new digit-like samples
- ✅ Well-organized latent space with digit clusters


## Usage

### Using Transformers

```python
from transformers import AutoModel
import torch
import torchvision.transforms as transforms

# Load model
model = AutoModel.from_pretrained("uday9k/Binarized_MNIST_VAE")

# Generate samples
with torch.no_grad():
    z = torch.randn(1, 20)  # Sample from prior
    generated = model.generate(z=z)
    # Reshape to image
    image = generated.view(28, 28).cpu().numpy()
```

### Visualizations Available

1. **Latent Space Visualization**: 2D scatter plot showing digit clusters
2. **Reconstructions**: Original vs. reconstructed digit comparisons  
3. **Generated Samples**: New digits sampled from the latent space
4. **Interpolations**: Smooth transitions between different digits
5. **Training Curves**: Loss components over training epochs

## Files and Outputs

- `MNIST_VAE_Train.ipynb`: Complete implementation with training and visualization
- `best_vae_model.pth`: Trained model weights
- `generated_samples`: Grid of generated digit samples as part of notebook
- `latent_space_visualization.png`: 2D latent space plot as part of notebook
- `reconstruction_comparison.png`: Original vs reconstructed images as part of notebook
- `latent_interpolation.png`: Interpolation between digit pairs as part of notebook
- `comprehensive_training_curves.png`: Training loss curves as part of notebook

## Applications

This VAE implementation can be used for:

- **Generative Modeling**: Create new handwritten digit images
- **Dimensionality Reduction**: Compress images to 2D representations
- **Anomaly Detection**: Identify unusual digits using reconstruction error
- **Data Augmentation**: Generate synthetic training data
- **Representation Learning**: Learn meaningful features for downstream tasks
- **Educational Purposes**: Understand VAE concepts and implementation

## Research and Educational Value

This implementation serves as an excellent educational resource for:

- Understanding Variational Autoencoders theory and practice
- Learning PyTorch implementation techniques
- Exploring generative modeling concepts
- Analyzing latent space representations
- Studying the balance between reconstruction and regularization

## Citation

If you use this implementation in your research or projects, please cite:

```bibtex
@misc{vae_mnist_implementation,
  title={Variational Autoencoder Implementation for MNIST},
  author={Uday Jain},
  year={2026},
  url={https://huggingface.co/uday9k/Binarized_MNIST_VAE}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Additional Resources

- **GitHub Repository**: [Profile](https://github.com/SpikeStriker/)

---

**Tags**: deep-learning, generative-ai, pytorch, vae, mnist, computer-vision, unsupervised-learning

**Model Card Authors**: Uday Jain 