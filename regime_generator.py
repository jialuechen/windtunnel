import torch
from models.vae_model import VAE

class RegimeGenerator:
    def __init__(self, vae_model_path, latent_dim):
        self.latent_dim = latent_dim
        self.model = VAE(input_dim=10, latent_dim=latent_dim)  # Adjust input_dim as needed
        self.model.load_state_dict(torch.load(vae_model_path))
        self.model.eval()

    def generate_regime(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            generated_data = self.model.decode(z)
        return generated_data.numpy()