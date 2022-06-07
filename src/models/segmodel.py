import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchmetrics import Accuracy, MetricCollection, JaccardIndex
import segmentation_models_pytorch as smp
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
from typing import Any, Tuple, Optional, Callable, cast

import pytorch_lightning as pl

from .unet import UNet


class SegModel(pl.LightningModule):
    """Semantic Segmentation Module.
    This is a basic semantic segmentation module implemented with Lightning.

    It uses the FCN ResNet50 model as an example.
    Adam optimizer is used along with Cosine Annealing learning rate scheduler.

    SegModel(
      (net): UNet(
        (layers): ModuleList(
          (0): DoubleConv(...)
          (1): Down(...)
          (2): Down(...)
          (3): Up(...)
          (4): Up(...)
          (5): Conv2d(64, 19, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    """

    def __init__(
        self,
        num_classes: int = 1,
        batch_size: int = 4,
        lr: float = 1e-3,
        num_layers: int = 3,
        features_start: int = 64,
        bilinear: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        self.net = UNet(
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

        self.lr = lr
        # self.loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.loss = nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_jacc = JaccardIndex(num_classes=2, ignore_index=0)
        self.val_jacc = JaccardIndex(num_classes=2, ignore_index=0)
        self.test_jacc = JaccardIndex(num_classes=2, ignore_index=0)

        self.save_hyperparameters()

        # self.train_metrics = MetricCollection(
        #     [
        #         JaccardIndex(
        #             num_classes=self.num_classes,
        #         ),
        #     ],
        #     prefix="train_",
        # )
        # self.val_metrics = self.train_metrics.clone(prefix="val_")
        # self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):

        preds, loss, gtruth = self._get_preds_loss_gtruth(batch)

        acc = self.train_acc(preds, gtruth)
        jaccard = self.train_jacc(preds, gtruth)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_accuracy", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_jaccard", self.train_jacc, on_step=False, on_epoch=True)

        return cast(Tensor, loss)

    def validation_step(self, batch, batch_idx):
        
        preds, loss, gtruth = self._get_preds_loss_gtruth(batch)

        acc = self.val_acc(preds, gtruth)
        jaccard = self.val_jacc(preds, gtruth)

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_accuracy", self.val_acc, on_step=False, on_epoch=True)
        self.log("val_jaccard", self.val_jacc, on_step=False, on_epoch=True)

        return preds

    def test_step(self, batch, batch_idx):

        preds, loss, gtruth = self._get_preds_loss_gtruth(batch)

        acc = self.test_acc(preds, gtruth)
        jaccard = self.test_jacc(preds, gtruth)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_accuracy", self.test_acc, on_step=False, on_epoch=True)
        self.log("val_jaccard", self.test_jacc, on_step=False, on_epoch=True)

        return preds
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        # predict dataloader loads only the inputs so batch=image here
        logits = self(batch)
        preds = torch.squeeze((logits.sigmoid() > 0.5).float(), dim=1)
        return preds

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt]
        # return [opt], [sch]

    def _get_preds_loss_gtruth(self, batch):

        image, mask = batch
        mask = mask.float()
        logits = self(image)

        loss = self.loss(logits, mask)

        preds = torch.squeeze((logits.sigmoid() > 0.5).float(), dim=1)
        # preds = torch.argmax(logits, dim=1)

        gtruth = torch.squeeze(mask, dim=1).int()

        return preds, loss, gtruth
