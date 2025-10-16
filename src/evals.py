import numpy as np
import torch
import torch.backends.cudnn as cudnn
import glob
import cv2
from pathlib import Path
from train import create_model_and_optimizer

if torch.cuda.is_available():
    device = torch.device("cuda")
    cudnn.benchmark = True
else:
  raise ValueError("GPU is not available")

def get_test_faces(channels_last = True):
    """
    Load a test dataset of faces, with 4 differente categories:
    \nLF: Light Female, LM: Light Male, DF: Dark Female and DM: Dark Male.
    Args:
        channels_last (bool) : If True, images are in (H, W, C) format. If False, images are in (C, H, W) format.
    Returns:
        Four images (dict): Four dictionaries for the 4 categories.
    """
    images = {"LF": [], "LM": [], "DF": [], "DM": []}
    for key in images.keys():
        FILES_PATH =  Path("data") / "faces" / key / "*.png"
        print("Loading test faces files from ", FILES_PATH)
        files = glob.glob(str(FILES_PATH))
        for file in sorted(files):
            image = cv2.resize(cv2.imread(file), (64,64))[:,:,::-1] / 255.0
            if not channels_last:
                image = np.transpose(image, (2,0,1))
            images[key].append(image)
    return images["LF"], images["LM"], images["DF"], images["DM"]

if not Path("checkpoints/dbvae_latest.pt").exists():
    raise FileNotFoundError("Checkpoint file not found")

test_faces = get_test_faces(channels_last=False)
keys = ["Light Female", "Light Male", "Dark Female", "Dark Male"]

#Load Checkpoint file of a trained DBVAE Model
ckpt = torch.load("checkpoints/dbvae_latest.pt", map_location="cpu")
model, opt = create_model_and_optimizer(latent_dim=ckpt["latent_dim"])
model.load_state_dict(ckpt["model_state_dict"])
opt.load_state_dict(ckpt["optimizer_state_dict"])
start_epoch = ckpt["epoch"]

dbvae_logits_list = []
with torch.inference_mode():
    for face in test_faces:
        face = torch.from_numpy(np.array(face, dtype=np.float32)).to(device)
        logits = model.predict(face)
        dbvae_logits_list.append(logits.detach().cpu().numpy()) #Prediction of each face

dbvae_logits_array = np.concatenate(dbvae_logits_list, axis=0)
dbvae_logits_tensor = torch.from_numpy(dbvae_logits_array)
dbvae_probs_tensor = torch.sigmoid(dbvae_logits_tensor)
dbvae_probs_array = dbvae_probs_tensor.squeeze(dim=-1).numpy()

xx = np.arange(len(keys))
dbvae_probs_mean = dbvae_probs_array.reshape(len(keys), -1).mean(axis=1)

for i in range(len(keys)):
    print(f"Test Sample of {keys[i]} got an Accuracy of {dbvae_probs_mean[i].item():.4f}")