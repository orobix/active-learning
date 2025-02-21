import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

# latent dimension for sampling


class VAE(nn.Module):
    """Calls the super init and defines the layers.

    Args:
        kernel_size: kernel height and width
        init_channels: number of the first filters
        image_channels: number of channels of the images
        latent_dim: goal dimensionality for latent representation
    """

    def __init__(self, kernel_size: int = 4, init_channels: int = 8, image_channels: int = 3, latent_dim: int = 3):
        super().__init__()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels * 2, kernel_size=kernel_size, stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels * 2, out_channels=init_channels * 4, kernel_size=kernel_size, stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels * 4, out_channels=64, kernel_size=kernel_size, stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels * 8, kernel_size=kernel_size, stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels * 8, out_channels=init_channels * 4, kernel_size=kernel_size, stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels * 4, out_channels=init_channels * 2, kernel_size=kernel_size, stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels * 2, out_channels=image_channels, kernel_size=kernel_size, stride=2, padding=1
        )

    def reparameterize(self, mu: float, log_var: float) -> torch.tensor:
        """Implements reparametrization trick, samples from the distribution without affecting backpropagation.

        Args:
            mu: mean from the encoder's latent space
            log_var: log variance from the encoder's latent space

        Returns:
            sample: extracted sample
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x: torch.Tensor) -> tuple[torch.tensor]:
        """Encoding layers.

        Args:
            x: Input tensor

        Returns:
            z: image's representation in the latent space
            mu: mean of the latent space
            log_var: log var of the latent space
        """
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def decode(self, z: torch.Tensor) -> torch.tensor:
        """Decoding layers.

        Returns:
            reconstructed image
        """
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)

        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction

    def forward(self, x: torch.Tensor) -> tuple[torch.tensor]:
        """Forward function of neural network, adds together encoding and decoding steps.

        Returns:
            reconstructed image
            mu
            log_var
        """
        z, mu, log_var = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var


def final_loss(bce_loss: float, mu: float, logvar: float) -> tuple[torch.tensor]:
    """Defines loss function as the reconstruction loss (BCELoss) plus the KL-Divergence.

    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        bce_loss: recontruction loss
        mu: the mean from the latent vector
        logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_vae(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module | None = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
) -> tuple[torch.tensor]:
    """Trains the VAE.

    Args:
        model: the VAE to be trained
        dataloader: images in batches
        criterion: loss criterion
        num_epochs (int): number of epochs for training
        learning_rate (float): learning rate
    """
    if criterion is None:
        criterion = nn.BCELoss(reduction="sum")

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, _ in dataloader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            bceloss = criterion(recon_x, x)
            loss = final_loss(bceloss, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    outputs = (x, recon_x)
    return outputs


def extract_latent_features(model: nn.Module, dataloader: DataLoader) -> torch.tensor:
    """Function to extract latent features from images using a VAE.

    Args:
        model: already trained VAE
        dataloader: images in bacthes

    Returns:
        latent_features: 3d features of the images extraceted by the VAE
    """
    model.eval()
    latent_features = []
    with torch.no_grad():
        for x, _ in dataloader:
            z, _, _ = model.encode(x)
            latent_features.append(z)
    latent_features = torch.cat(latent_features, dim=0)
    return latent_features


class VAE224(nn.Module):
    """class for variational autoencoder, modified to work with 224x224 images."""

    def __init__(self, kernel_size: int = 4, init_channels: int = 8, image_channels: int = 3, latent_dim: int = 3):
        """Calls the super init and defines the layers.

        Args:
            kernel_size: kernel height and width
            init_channels: number of the first filters
            image_channels: number of channels of the images
            latent_dim: goal dimensionality for latent representation
        """
        super().__init__()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels * 2, kernel_size=kernel_size, stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels * 2, out_channels=init_channels * 4, kernel_size=kernel_size, stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels * 4, out_channels=64, kernel_size=kernel_size, stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=init_channels * 8,
            kernel_size=kernel_size,
            stride=4,
            padding=0,
            output_padding=0,
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels * 8,
            out_channels=init_channels * 4,
            kernel_size=kernel_size,
            stride=4,
            padding=0,
            output_padding=2,
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels * 4,
            out_channels=init_channels * 2,
            kernel_size=kernel_size,
            stride=4,
            padding=0,
            output_padding=2,
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels * 2,
            out_channels=image_channels,
            kernel_size=kernel_size,
            stride=3,
            padding=0,
            output_padding=1,
        )

    def reparameterize(self, mu: float, log_var: float) -> torch.tensor:
        """Implements reparametrization trick, samples from the distribution without affecting backpropagation.

        Args:
            mu: mean from the encoder's latent space
            log_var: log variance from the encoder's latent space

        Returns:
            sample: extracted sample
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x: torch.Tensor) -> tuple[torch.tensor]:
        """Encoding layers.

        Returns:
            z: image's representation in the latent space
            mu: mean of the latent space
            log_var: log var of the latent space
        """
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def decode(self, z: torch.Tensor) -> torch.tensor:
        """Decoding layers.

        Returns:
            reconstructed image
        """
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)

        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction

    def forward(self, x: torch.Tensor) -> tuple[torch.tensor]:
        """Forward function of neural network, adds together encoding and decoding steps.

        Returns:
            reconstructed image
            mu
            log_var
        """
        z, mu, log_var = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
