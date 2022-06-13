import os
import numpy as np
import torch

from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from ..datasets import TreeSegments
from typing import Any, Optional


class TreeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        target_dir: str = "./",
        test_data_dir: str = "./",
        test_target_dir: str = "./",
        batch_size: int = 8,
        num_workers: int = 0,
        drop_last_batch: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.test_data_dir = test_data_dir
        self.test_target_dir = test_target_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([3.5133], [1.6922])]
        )
        self.target_transform = transforms.ToTensor()
        self.drop_last = drop_last_batch

    def setup(self, stage: Optional[str] = None):

        validation_split = 0.8

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full_dataset = TreeSegments(
                self.data_dir,
                self.target_dir,
                train=True,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.dataset_size = len(full_dataset)

            # this is used when debugging with one sample
            if self.dataset_size == 1:
                self.train_dataset = full_dataset
                self.val_dataset = full_dataset
            else:
                train_size = int(0.8 * self.dataset_size)
                val_size = self.dataset_size - train_size
                self.train_dataset, self.val_dataset = random_split(
                    full_dataset, [train_size, val_size]
                )

                # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = TreeSegments(
                self.test_data_dir,
                self.test_target_dir,
                train=True,  # this is a hack until I sort out predict and test dls
                transform=self.transform,
                target_transform=self.target_transform,
            )

        if stage == "predict" or stage is None:
            self.predict_dataset = TreeSegments(
                self.data_dir,
                self.target_dir,
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)
