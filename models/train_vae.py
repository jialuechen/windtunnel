import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from vae_model import VAE

def train_vae(data, input_dim, latent_dim, epochs=50, batch_size=32, learning_rate=0.001):
    # Prepare data
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    reconstruction_loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(x)

            # Compute losses
            reconstruction_loss = reconstruction_loss_fn(reconstructed, x)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + kl_divergence

            # Backpropagation
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    return model