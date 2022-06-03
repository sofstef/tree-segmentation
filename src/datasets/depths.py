import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, Dataset, DataLoader
from typing import Any, Tuple, Optional, Callable
import glob


class TreeSegments(Dataset):
    """Class defining procedure to load ARCore depth maps"""

    def __init__(
        self,
        data_dir: str,
        target_dir: str,
        train: bool = True,
        img_size: tuple = (160, 120),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # Initialize list of relative path to each image
        self.data_paths = [
            os.path.join(file_name) for file_name in self.listdir_nohidden(data_dir)
        ]
        self.target_paths = [
            os.path.join(file_name) for file_name in self.listdir_nohidden(target_dir)
        ]
        self.train = train
        self.img_size = img_size
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:

        data_path = self.data_paths[idx]
        img = np.loadtxt(data_path, delimiter=",", usecols=range(0, 3))
        img = img[:, 2].reshape(120, 160)

        if self.transform is not None:
            img = self.transform(img)

        if self.train == True:
            # find matching mask – a bit messy but should work
            # mask_path = self.mask_paths[idx]
            sample_id = data_path.split("/")[-1].split("_")[-2:]
            sample_id = "_".join(sample_id)

            for m in self.target_paths:
                if sample_id in m:
                    target_path = m

            mask = Image.open(target_path).convert("L")
            mask = np.array(mask.resize(self.img_size))

            if self.target_transform is not None:
                mask = self.target_transform(mask).type(torch.int64)

            return img, mask

        else:
            # this is for prediction step, not sure if best solution
            return img

    def __len__(self):
        return len(self.data_paths)

    def listdir_nohidden(self, path):
        return glob.glob(os.path.join(path, "*"))
