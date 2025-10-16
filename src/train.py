import sys
from pathlib import Path

# Make the project root importable when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.datasetLoader import TrainDatasetLoader
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model import DB_VAE, debiasing_loss_function

import argparse

#Constants
CACHE_DIR = Path.cwd() / ".cache"

# Check if using a GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    cudnn.benchmark = True
else:
  raise ValueError("GPU is not available")

def get_loader_and_faces():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path_to_training_data = CACHE_DIR.joinpath("train_face.h5")
    if path_to_training_data.is_file():
        print(f"Using cached training data from {path_to_training_data}")
    else:
        print(f"Downloading training data to {path_to_training_data}")
        url = "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
        torch.hub.download_url_to_file(url, str(path_to_training_data))
    loader = TrainDatasetLoader(str(path_to_training_data), channels_last=False)
    all_faces = loader.get_all_train_faces()
    return loader, all_faces

def get_latent_mu(images, dbvae, batch_size=64):
    """
    Get the latent mean vectors for a set of images using the DB-VAE model.
    Args:
        images (np.ndarray): Input images.
        dbvae (DB_VAE): Debiased VAE model.
        batch_size (int): Batch size for processing images.
    Returns:
        mu (np.ndarray): Latent mean vectors for the input images.
    """
    dbvae.eval()
    all_z_mean = []
    images_t = torch.from_numpy(images).float()

    with torch.inference_mode():
        for start in range(0, len(images_t), batch_size):
            end = start + batch_size
            batch = images_t[start:end].to(device).permute(0, 3, 1, 2)
            _, z_mean, _, _ = dbvae(batch)
            all_z_mean.append(z_mean.detach().cpu())
    
    print("Number of batches:", len(all_z_mean))
    z_mean_full = torch.cat(all_z_mean, dim=0) #Ver isto
    mu = z_mean_full.numpy()
    return mu

def get_training_sample_prob(images, dbvae, batch_size, bins=10, smoothing_fac=0.001):
    """
    Calculates the probability chance of images under represented in the latent variables
    distribution, favoring images who are rarer in the distribution.
    Args:
        images (np.ndarray): Input images sample.
        dbvae (DB_VAE): Debiased VAE model.
        batch_size (int): Batch size for processing images.
        bins (int): Number of histogram intervals used to estimate the latent variable distribution along each dimension.
        smoothing_fac (float): Small float value to avoid zeros
    Returns:
        training_sample_p (np.ndarray): Sample probabilities where rare samples in the distribution have a high probability 
    """
    print("Recomputing the sample probabilities")

    mu = get_latent_mu(images, dbvae, batch_size)
    training_sample_p = np.zeros(mu.shape[0], dtype=np.float64)

    for i in range(dbvae.latent_dim):
        latent_distribution = mu[:, i]

        hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)

        bin_edges[0] = -float("inf")
        bin_edges[-1] = float("inf")

        bin_idx = np.digitize(latent_distribution, bin_edges)

        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density /= np.sum(hist_smoothed_density)

        p = 1.0 / (hist_smoothed_density[bin_idx -1])

        p /= np.sum(p)

        training_sample_p = np.maximum(training_sample_p, p)
    
    training_sample_p /= np.sum(training_sample_p)
    return training_sample_p

def debiasing_train_step(x, y, dbvae, optimizer):
    """
    Single training step for the DB-VAE model.
    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): True labels for the input images.
        dbvae (DB_VAE): Model.
        optimizer (torch.optim.Optimizer): Optimizer.
    Returns:
        loss (torch.Tensor): Computed loss for the training step.
    """
    optimizer.zero_grad()
    y_logit, z_mean, z_logsigma, x_recon = dbvae(x)
    loss, class_loss = debiasing_loss_function(x, x_recon, y, y_logit, z_mean, z_logsigma)
    loss.backward()
    optimizer.step()
    return loss

def create_model_and_optimizer(latent_dim=128, learning_rate=5e-4):
    dbvae = DB_VAE(latent_dim).to(device)
    optimizer = optim.Adam(dbvae.parameters(), lr=learning_rate)
    return dbvae, optimizer

def save_checkpoint(dbvae, optimizer, epoch, output_dir):
    """
    Save model/optimizer state so training can be resumed later.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": dbvae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "latent_dim": dbvae.latent_dim,
    }
    torch.save(ckpt, output_dir / f"dbvae_epoch_{epoch}.pt")
    torch.save(ckpt, output_dir / "dbvae_latest.pt")

def run_training(num_epochs=2, batch_size=64, learning_rate=5e-4, latent_dim=128, bins=10, smoothing_fac=0.001, output_dir="checkpoints", save_every=1):
    loader, all_faces = get_loader_and_faces()
    dbvae, optimizer = create_model_and_optimizer(latent_dim=latent_dim, learning_rate=learning_rate)

    step = 0
    for i in range(num_epochs):
        print(f"Starting epoch {i+1}/{num_epochs}")
        p_faces = get_training_sample_prob(all_faces, dbvae, batch_size, bins=bins, smoothing_fac=smoothing_fac)

        for _ in tqdm(range(len(loader)//batch_size)):
            x, y = loader.get_batch(batch_size, p_pos=p_faces)
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)

            loss = debiasing_train_step(x, y, dbvae, optimizer)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
              
            step += 1
        if ((i + 1) % save_every) == 0:
            save_checkpoint(dbvae, optimizer, epoch=i + 1, output_dir=output_dir)

def main():
    parser = argparse.ArgumentParser(description="Train DB-VAE")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--bins", type=int, default=10, help="Histogram bins for debiasing")
    parser.add_argument("--smoothing-fac", type=float, default=0.001, help="Histogram smoothing factor")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    run_training(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        bins=args.bins,
        smoothing_fac=args.smoothing_fac,
        output_dir=args.output_dir,
        save_every=args.save_every,
    )

if __name__ == "__main__":
    main()