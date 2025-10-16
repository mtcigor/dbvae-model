import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn.functional as F

#Verify if CUDA is available for the machine the torch is running
if torch.cuda.is_available():
    device = torch.device("cuda")
    cudnn.benchmark = True
else:
  raise ValueError("GPU is not available")

def make_standard_classifier(n_outputs, in_channels=3, H=64, W=64, n_filters=12):
    """
    Create a standard CNN classifier model.
    Args:
        n_outputs (int): Number of output classes.
    Returns:
        model (nn.Module): The constructed CNN model.
    """
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x
    
    model = nn.Sequential(
        ConvBlock(in_channels, n_filters, kernel_size=5, stride=2, padding=2),
        ConvBlock(n_filters, 2*n_filters, kernel_size=5, stride=2, padding=2),
        ConvBlock(2*n_filters, 4*n_filters, kernel_size=3, stride=2, padding=1),
        ConvBlock(4*n_filters, 8*n_filters, kernel_size=3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(H // 16 * W // 16 * 8 * n_filters, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, n_outputs),
        )
    return model.to(device)

def vae_loss_function(x, x_recon, mu, logsima, kl_weight=0.0005):
    """
    VAE Loss Function
        Computes the loss for the Variational Autoencoder (VAE) model.
        The loss is a combination of the reconstruction loss and the KL divergence loss.
        Args:
            x (torch.Tensor): Original input images.
            x_recon (torch.Tensor): Reconstructed images from the VAE.
            mu (torch.Tensor): Mean of the latent variable distribution.
            logsima (torch.Tensor): Log variance of the latent variable distribution.
            kl_weight (float): Weight for the KL divergence loss term.
        Returns:
            torch.Tensor: Total loss (reconstruction + KL divergence).

    """
    latent_loss = 0.5 * torch.sum(torch.exp(logsima) + mu**2 - 1.0 - logsima)
    reconstruction_loss = torch.mean(torch.abs(x - x_recon  ))
    return kl_weight * latent_loss + reconstruction_loss

def sampling_reparameterization(mu, logsima):
    """
    Reparameterization trick to sample from N(mu, sigma^2) from N(0,1) (Gaussian distribution).
    Args:
        mu (torch.Tensor): Mean of the latent variable distribution.
        logsima (torch.Tensor): Log variance of the latent variable distribution.
    Returns:
        torch.Tensor: Sampled latent variable.
    """
    eps = torch.randn_like(mu)
    sigma = torch.exp(logsima)
    return mu + sigma * eps

def debiasing_loss_function(x, x_pred, y, y_logits, mu, logsigma, kl_weight=0.0005):
    """
    DV-VAE Loss function
        Computes the loss function for the Debiased Variation Autoencoder Model.
        The total loss is the mean combination of the classification loss and the VAE loss if the classification true result is a face.
        Args:
            x (torch.Tensor): Original input images.
            x_pred (torch.Tensor): Reconstructed images from the VAE.
            y (torch.Tensor): True labels for the input images.
            y_logits (torch.Tensor): Predicted logits from the classifier.
            mu (torch.Tensor): Mean of the latent variable distribution.
            logsigma (torch.Tensor): Log variance of the latent variable distribution.
            kl_weight (float): Weight for the KL divergence loss term.
        Returns:
            torch.Tensor: Total loss (classification + VAE loss for faces).
            torch.tensor: Classification loss.
    """
    vae_loss = vae_loss_function(x, x_pred, mu, logsigma, kl_weight)
    classification_loss = F.binary_cross_entropy_with_logits(y_logits, y, reduction='none')

    #Which training data are images of faces
    y.float()
    face_indicator = (y == 1.0).float()

    total_loss = torch.mean(classification_loss * face_indicator + vae_loss)

    return total_loss, classification_loss

def make_face_decoder_network(latent_dim=128, n_filters=12):
    """
    Decoder network of a VAE model.
    Args:
        latent_dim (int): Dimension of the latent space.
        n_filters (int): Number of filters in the convolutional layers.
    Returns:
        FaceDecoder (nn.Module): Decoder network model.
    """
    class FaceDecoder(nn.Module):
        def __init__(self, latent_dim, n_filters):
            super(FaceDecoder, self).__init__()
            self.latent_dim = latent_dim
            self.n_filters = n_filters
            self.linear = nn.Sequential(nn.Linear(latent_dim, 8 * self.n_filters * 4 * 4), nn.ReLU())
        
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=8 * self.n_filters,
                    out_channels=4 * self.n_filters,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_channels=4 * self.n_filters,
                    out_channels=2 * self.n_filters,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_channels=2 * self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    in_channels=self.n_filters,
                    out_channels=3,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
            )
        def forward(self, z):
            x = self.linear(z)
            x = x.view(-1, 8*self.n_filters, 4, 4)
            x = self.deconv(x)
            return x
    return FaceDecoder(latent_dim, n_filters)

class DB_VAE(nn.Module):
    """
    Debiased Variational Autoencoder Model.
    Args:
        latent_dim (int): Dimension of the latent space.
    Returns:
        DB_VAE (nn.Module): Debiased VAE model.
    """
    def __init__(self, latent_dim=128, n_filters=12, n_outputs=1, in_channels=3, H=64, W=64):
        super(DB_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_filters = n_filters
        self.n_outputs = n_outputs
        self.in_channels = in_channels
        self.H = H
        self.W = W
        self.encoder = make_standard_classifier(2 * latent_dim+1, in_channels, H, W, n_filters)
        self.decoder = make_face_decoder_network(latent_dim=latent_dim)

    def encode(self, x):
        encoder_out = self.encoder(x)

        y_logit = encoder_out[:, 0].unsqueeze(-1)
        z_mu = encoder_out[:, 1 : self.latent_dim + 1]
        z_logsigma = encoder_out[:, self.latent_dim + 1 :]
        return y_logit, z_mu, z_logsigma
    
    def reparameterize(self, z_mu, z_logsigma):
        return sampling_reparameterization(z_mu, z_logsigma)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        y_logit, z_mu, z_logsigma = self.encode(x)
        z = self.reparameterize(z_mu, z_logsigma)
        recon = self.decode(z)
        return y_logit, z_mu, z_logsigma, recon
    
    def predict(self, x):
        y_logit, _, _ = self.encode(x)
        return y_logit