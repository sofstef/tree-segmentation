import os
import numpy as np
import torch

from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from ..datasets import TreeSegments
from typing import Any, Optional

class TreeDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "./",
                 target_dir: str = "./",
                 batch_size: int = 8,
                ):
        super().__init__()
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(),])
                                             # transforms.Normalize([0.2860, 0.4530, 0.4528],
                                             #                      [0.1852, 0.2681, 0.1247])])
            
    def setup(self, stage: Optional[str] = None):
        
        validation_split = .8
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full_dataset = TreeSegments(self.data_dir,
                                      self.target_dir,
                                      train=True,
                                      transform=self.transform,
                              )
            self.dataset_size = len(full_dataset)
            
            # this is used when debugging with one sample
            if self.dataset_size == 1: 
                self.train_dataset = full_dataset
                self.val_dataset = full_dataset 
            else:
                train_size = int(0.8 * self.dataset_size)
                val_size = self.dataset_size - train_size
                self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        
                # Assign test dataset for use in dataloader(s)
#         if stage == "test" or stage is None:
#             self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.predict_dataset = TreeSegments(self.data_dir,
                                      self.target_dir,
                                      train=False,
                                      transform=self.transform,
                              )
        
        
    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          # drop_last=True,
                         )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          # drop_last=True,
                         )

#     def test_dataloader(self) -> DataLoader[Any]:
#         return DataLoader(self.tree_test, batch_size=32)

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.predict_dataset, 
                          batch_size=self.batch_size)