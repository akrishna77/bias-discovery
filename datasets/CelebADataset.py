import os

from torch.utils.data import Dataset
import torch
import pandas as pd

from PIL import Image


class CelebADataset(Dataset):
    """CelebA dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attributes = None
        self.csv_file = csv_file

        self.attributes = pd.read_csv(csv_file, delimiter="\s+", engine="python", header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = str(self.attributes.iloc[idx, 0])
        img_path = os.path.join(str(self.root_dir), img_name)

        attr = self.attributes.iloc[idx, self.attributes.columns.get_loc("Smiling")]
        if attr == -1:
            attr = 0
        sample = {"image_name": img_name, "image_path": img_path, "image": Image.open(img_path), "attributes": attr}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample
