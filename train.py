"""
Description: Runs binary segmentation experiment

Usage: train.py [options] --cfg=<path_to_config>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.

"""


import os
import yaml
from docopt import docopt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.datasets import TreeSegments

from src.datamodules import TreeDataModule
from src.models import SegModel
from src.evaluation import LogPredictionSamplesCallback


def main(conf):

    datamodule = TreeDataModule(
        data_dir=conf["datamodule"]["data_dir"],
        target_dir=conf["datamodule"]["target_dir"],
        batch_size=conf["datamodule"]["batch_size"],
        num_workers=conf["datamodule"]["num_workers"],
    )

    model = SegModel(
        in_channels=conf["module"]["in_channels"],
        encoder_name=conf["module"]["encoder_name"],
        encoder_weights=conf["module"]["encoder_weights"],
        num_classes=conf["module"]["num_classes"],
        loss=conf["module"]["loss"],
        ignore_zeros=conf["module"]["ignore_zeros"],
        lr=conf["module"]["learning_rate"],
        jaccard_average=conf["module"]["jaccard_average"],
        learning_rate_schedule_patience=conf["module"][
            "learning_rate_schedule_patience"
        ],
    )

    wandb_logger = WandbLogger(
        project="Trees",
        log_model="all",
        name=conf["logger"]["run_name"],
        save_code=False,
        save_dir=conf["logger"]["log_dir"],
    )

    callbacks = [
        LogPredictionSamplesCallback(),
        ModelCheckpoint(monitor="val_accuracy", mode="max"),
        ModelCheckpoint(monitor="val_jaccard", mode="max"),
        ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=conf["trainer"]["save_top_k"]
        ),
        ModelCheckpoint(monitor="val_f1", mode="max"),
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=10,
            mode="min",
            verbose=False,
        ),
    ]

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=conf["trainer"]["fast_dev_run"],
        default_root_dir=conf["trainer"]["default_root_dir"],
        gpus=conf["trainer"]["gpus"],
        max_epochs=conf["trainer"]["max_epochs"],
        log_every_n_steps=conf["trainer"]["log_every_n_steps"],
        # precision=conf["trainer"]["precision"],
        auto_lr_find=conf["trainer"]["auto_lr_find"],
    )

    wandb_logger.watch(model)

    if conf["trainer"]["auto_lr_find"]:
        trainer.tune(model, datamodule)

    trainer.fit(model, datamodule)


if __name__ == "__main__":

    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args["--cfg"], "r") as f:
        cfg = yaml.safe_load(f)

    # set random seed for reproducibility
    pl.seed_everything(cfg["program"]["seed"])

    # TRAIN
    main(cfg)
