"""
    Modified code from original source:
    © MIT Introduction to Deep Learning
    http://introtodeeplearning.com
"""

from torch.utils.data import Dataset
import h5py
import torch
import numpy as np

class TrainDatasetLoader(Dataset):
    """
    Dataset loader for training data.
    Args:
        data_path (str): Path to the HDF5 file containing the dataset.
        channels_last (bool): If True, images are in (H, W, C) format. If False, images are in (C, H, W) format.
    Attributes:
        images (np.ndarray): Array of images loaded from the HDF5 file.
        labels (np.ndarray): Array of labels loaded from the HDF5 file.
        train_inds (np.ndarray): Array of shuffled indices for training samples.
        pos_train_inds (np.ndarray): Array of indices for positive samples.
        neg_train_inds (np.ndarray): Array of indices for negative samples.
    Methods:
        __len__: Returns the number of training samples.
        __getitem__: Retrieves an image and its corresponding label by index.
        get_train_steps_per_epoch: Calculates the number of training steps per epoch.
        get_batch: Retrieves a batch of images and labels, with options for positive-only samples and returning indices.
        get_n_most_prob_faces: Retrieves the n most probable positive samples based on provided probabilities.
        get_all_train_faces: Retrieves all positive training samples.
    """
    def __init__(self, data_path, channels_last=False):
        print(f"Opening {data_path}")
        self.cache = h5py.File(data_path, "r")
        self.images = self.cache["images"][:]
        self.labels = self.cache["labels"][:].astype(np.float32)
        self.channels_last = channels_last
        self.image_dims = self.images.shape

        n_train_samples = self.image_dims[0]
        #Array of n_train_samples shuffled randomly
        self.train_inds = np.random.permutation(np.arange(n_train_samples))
        self.pos_train_inds = self.train_inds[self.labels[self.train_inds, 0] == 1.0]
        self.neg_train_inds = self.train_inds[self.labels[self.train_inds, 0] != 1.0]

    def __len__(self):
        return len(self.train_inds)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # normalize to [0,1]
        img = (img[:, :, ::-1] / 255.0).astype(np.float32)

        if not self.channels_last:  # convert to [H, W, C] to [C,H,W]
            img = np.transpose(img, (2,0,1))

        return torch.tensor(img), torch.tensor(label)

    def get_train_steps_per_epoch(self, batch_size, factor=10):
        return self.__len__() // factor // batch_size

    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None, return_inds=False):
        if only_faces:
            select_inds = np.random.choice(
                self.pos_train_inds, size=n, replace=False, p=p_pos)
        else:
            selected_pos_inds = np.random.choice(
                self.pos_train_inds, size=n//2, replace=False, p=p_pos)
            selected_neg_inds = np.random.choice(
                self.neg_train_inds, size=n//2, replace=False, p=p_neg)
            selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))

        sorted_inds = np.sort(selected_inds)
        train_img = (self.images[sorted_inds, :,:, ::-1] / 255.0).astype(np.float32)
        train_label = self.labels[sorted_inds, ...]

        if not self.channels_last:
            train_img = np.ascontiguousarray(
                np.transpose(train_img, (0,3,1,2)))
        return (
            (train_img, train_label, sorted_inds)
            if return_inds
            else (train_img, train_label))

    def get_n_most_prob_faces(self, prob, n):
        """
        From the positive training set, sort by probability, look at the top 10n, 
        take every 10th one to get n images, normalize them, and return them.
        """
        idx = np.argsort(prob)[::-1]
        most_prob_inds = self.pos_train_inds[idx[: 10 * n : 10]]
        return (self.images[most_prob_inds, ...] / 255.0).astype(np.float32)

    def get_all_train_faces(self):
        return self.images[self.pos_train_inds]