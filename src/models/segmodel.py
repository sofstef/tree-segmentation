import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchmetrics import Accuracy, JaccardIndex, F1Score, MeanSquaredError
import segmentation_models_pytorch as smp
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss
from typing import Any, Tuple, Optional, Callable, Dict, cast

import pytorch_lightning as pl


class SegModel(pl.LightningModule):
    """Semantic Segmentation task implemeneted with Lightning."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.net = smp.Unet(
            encoder_name=self.hyperparams["encoder_name"],
            encoder_weights=self.hyperparams["encoder_weights"],
            in_channels=self.hyperparams["in_channels"],
            classes=self.hyperparams["num_classes"],
        )

        if self.hyperparams["loss"] == "dice":
            self.loss = smp.losses.DiceLoss(
                smp.losses.BINARY_MODE,
                from_logits=True,
                ignore_index=self.hyperparams["ignore_zeros"],
            )

        elif self.hyperparams["loss"] == "softBCE":
            self.loss = smp.losses.SoftBCEWithLogitsLoss(
                ignore_index=0,
            )

        else:
            raise ValueError(f"Loss type '{self.hyperparams['loss']}' is not valid.")

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.ignore_zeros = None if kwargs["ignore_zeros"] else 0

        self.config_task()

        self.train_acc = Accuracy(num_classes=1, multiclass=False)
        self.val_acc = Accuracy(num_classes=1, multiclass=False)
        self.test_acc = Accuracy(num_classes=1, multiclass=False)
        self.train_jacc = JaccardIndex(2, average=self.hyperparams["jaccard_average"])
        self.val_jacc = JaccardIndex(2, average=self.hyperparams["jaccard_average"])
        self.test_jacc = JaccardIndex(2, average=self.hyperparams["jaccard_average"])
        self.train_f1 = F1Score(num_classes=1, multiclass=False)
        self.val_f1 = F1Score(num_classes=1, multiclass=False)
        self.test_f1 = F1Score(num_classes=1, multiclass=False)
        self.rmse = MeanSquaredError()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):

        preds, loss, gtruth = self._get_preds_loss_gtruth(batch)

        acc = self.train_acc(preds, gtruth)
        jaccard = self.train_jacc(preds, gtruth)
        f1 = self.train_f1(preds, gtruth)

        # TO DO: want to pass object and not result but this is all too clunky
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_accuracy", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_jaccard", self.train_jacc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)

        return cast(Tensor, loss)

    def validation_step(self, batch, batch_idx):

        preds, loss, gtruth = self._get_preds_loss_gtruth(batch)

        acc = self.val_acc(preds, gtruth)
        jaccard = self.val_jacc(preds, gtruth)
        f1 = self.val_f1(preds, gtruth)

        metrics = {"val_acc": acc, "val_loss": loss, "val_jacc": jaccard, "val_f1": f1}

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_accuracy", self.val_acc, on_step=False, on_epoch=True)
        self.log("val_jaccard", self.val_jacc, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

        return metrics

    def test_step(self, batch, batch_idx):

        self.test_preds, loss, self.test_gtruth = self._get_preds_loss_gtruth(batch)

        acc = self.test_acc(self.test_preds, self.test_gtruth)
        jaccard = self.test_jacc(self.test_preds, self.test_gtruth)
        f1 = self.test_f1(self.test_preds, self.test_gtruth)
        rmse = self.rmse(self.test_preds, self.test_gtruth)

        metrics = {
            "test_acc": acc,
            "test_loss": loss,
            "test_jacc": jaccard,
            "test_f1": f1,
            "test_rmse": rmse,
        }

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_jaccard", self.test_jacc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_rmse", self.rmse, on_step=False, on_epoch=True)

        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        # predict dataloader loads only the inputs so batch=image here
        logits = self(batch)
        preds = torch.squeeze((logits.sigmoid() > 0.5).int(), dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hyperparams["lr"])
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": ReduceLROnPlateau(
            #         optimizer,
            #         patience=self.hyperparams["learning_rate_schedule_patience"],
            #     ),
            #     "monitor": "val_loss",
            #     },
        }

    def _get_preds_loss_gtruth(self, batch):

        image, mask = batch
        mask = mask.float()
        logits = self(image)

        loss = self.loss(logits, mask)

        # preds being int needed for correct metric computation
        # make sure there's no reason why we'd want them as floats
        preds = torch.squeeze((logits.sigmoid() > 0.5).int(), dim=1)
        gtruth = torch.squeeze(mask, dim=1).int()

        return preds, loss, gtruth
