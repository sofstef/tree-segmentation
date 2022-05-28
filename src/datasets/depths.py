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
    """Class defining procedure to load ARCore depth maps
    """
    
    def __init__(self,
                 depth_dir: str,
                 mask_dir: str,
                 train: bool = True,
                 img_size: tuple = (160, 120),
                 transform: Optional[Callable] = None,) -> None:
        # Initialize list of relative path to each image
        self.depth_paths = [os.path.join(file_name) for file_name in self.listdir_nohidden(depth_dir)]
        self.mask_paths = [os.path.join(file_name) for file_name in self.listdir_nohidden(mask_dir)]
        self.train = train
        self.img_size = img_size
        self.transform = transform
        self.target_transform = transforms.ToTensor()
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        
        depth_path = self.depth_paths[idx]
        img = Image.open(depth_path)
        
        # find matching mask – a bit messy but should work
        # mask_path = self.mask_paths[idx]
        sample_id = depth_path.split('/')[-1].split('.')[0].split("_")[-2:]
        sample_id = "_".join(sample_id)
        
        for m in self.mask_paths:
            if sample_id in m:
                mask_path = m
                
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.img_size)

        if self.transform is not None:
            img = self.transform(img)
        
        mask = self.target_transform(mask).type(torch.int64)
        return img, mask
        # return {
        #     "image": img,
        #     "mask": mask,
        # }
    
    def __len__(self):
        return len(self.depth_paths)
    
        
    def listdir_nohidden(self, path):
        return glob.glob(os.path.join(path, '*'))