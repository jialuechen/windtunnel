#### **`docs/api_reference.md`**
```md
# API Reference

## `windtunnel.RegimeGenerator`
```python
class RegimeGenerator:
    def __init__(self, vae_model_path, latent_dim):
        """
        Initializes the RegimeGenerator with a pre-trained VAE model and latent dimension.

        Args:
            vae_model_path (str): Path to the pre-trained VAE model.
            latent_dim (int): Dimensionality of the latent space.
        """
        pass

    def generate_regime(self, num_samples):
        """
        Generates a specified number of regimes using the VAE model.

        Args:
            num_samples (int): Number of regimes to generate.

        Returns:
            list: A list of generated regimes.
        """
        pass
```