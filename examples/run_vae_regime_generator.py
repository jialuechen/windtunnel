import numpy as np
from models.train_vae import train_vae
from genmarket.regime_generator import RegimeGenerator

# Step 1: Generate synthetic data for training
synthetic_data = np.random.rand(1000, 10)  # 1000 samples, 10 features

# Step 2: Train the VAE model
vae_model = train_vae(data=synthetic_data, input_dim=10, latent_dim=5, epochs=20)

# Step 3: Save the trained model
torch.save(vae_model.state_dict(), "vae_model.pth")

# Step 4: Use the trained VAE in RegimeGenerator
regime_generator = RegimeGenerator(vae_model_path="vae_model.pth", latent_dim=5)
generated_regimes = regime_generator.generate_regime(num_samples=10)

print("Generated Market Regimes:")
print(generated_regimes)