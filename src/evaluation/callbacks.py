import torch
import wandb
from pytorch_lightning.callbacks import Callback

### TO DO: change to log multiple samples in a batch


class LogPredictionSamplesCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # Dict of possible classes
        class_labels = {
            0: "other",
            1: "tree",
        }

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        n = 0
        if batch_idx == 0:
            x, y = batch

            image = x[n]
            ground_truth = torch.squeeze(y[n].detach()).numpy()
            prediction_mask = torch.squeeze(outputs[n]).detach().numpy()

            wandb.log(
                {
                    "Depth - Prediction - Ground Truth": wandb.Image(
                        image[0, :, :],
                        masks={
                            "predictions": {
                                "mask_data": prediction_mask,
                                "class_labels": class_labels,
                            },
                            "ground_truth": {
                                "mask_data": ground_truth,
                                "class_labels": class_labels,
                            },
                        },
                    )
                }
            )
