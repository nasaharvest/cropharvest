import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from pathlib import Path
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score

from .lstm import Classifier

from cropharvest.datasets import CropHarvest


class Pretrainer(pl.LightningModule):
    def __init__(
        self,
        root,
        batch_size: int,
        learning_rate: float,
        classifier_vector_size: int,
        classifier_dropout: float,
        classifier_base_layers: int,
        num_classification_layers: int,
        pretrained_val_ratio: float,
        model_name: str,
    ) -> None:
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrained_val_ratio = pretrained_val_ratio
        self.model_name = model_name

        (Path(self.root) / self.model_name).mkdir(exist_ok=True)

        self.classifier = Classifier(
            input_size=self.get_dataset(is_val=False).num_bands,
            classifier_vector_size=classifier_vector_size,
            classifier_dropout=classifier_dropout,
            classifier_base_layers=classifier_base_layers,
            num_classification_layers=num_classification_layers,
        )

        self.best_val_loss: float = np.inf

        self.loss = nn.BCELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

    def get_dataset(self, is_val: bool) -> CropHarvest:
        return CropHarvest(
            self.root, download=True, is_val=is_val, val_ratio=self.pretrained_val_ratio
        )

    def train_dataloader(self):
        return DataLoader(self.get_dataset(is_val=False), shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.get_dataset(is_val=True), batch_size=self.batch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        preds = self.forward(x.float())
        loss = self.loss(preds.squeeze(1), y.float())

        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        preds = self.forward(x.float())
        loss = self.loss(preds.squeeze(1), y.float())

        return {"val_loss": loss, "log": {"val_loss": loss}, "preds": preds, "labels": y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()

        if len(np.unique(labels)) == 1:
            # this happens during the sanity check
            return {
                "val_loss": avg_loss,
            }

        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_auc_roc": roc_auc_score(labels, preds),
            "val_accuracy": accuracy_score(labels, preds > 0.5),
        }

        if float(avg_loss) < self.best_val_loss:
            self.best_val_loss = float(avg_loss)
            print("Saving best state_dict")
            self.save_state_dict()

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def save_state_dict(self) -> None:
        torch.save(
            self.classifier.state_dict(), Path(self.root) / self.model_name / "state_dict.pth"
        )


def pretrain_model(
    root,
    classifier_vector_size: int,
    classifier_dropout: float,
    classifier_base_layers: int,
    num_classification_layers: int,
    model_name: str,
    pretrained_val_ratio: float = 0.1,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    max_epochs: int = 1000,
    patience: int = 10,
) -> Classifier:
    r"""
    Initialize and pretrain a classifier on a global crop vs. non crop task

    :root: The path to the data
    :param classifier_vector_size: The LSTM hidden vector size to use
    :param classifier_dropout: The value for variational dropout between LSTM timesteps to us
    :param classifier_base_layers: The number of LSTM layers to use
    :param num_classification_layers: The number of linear classification layers to use on top
        of the LSTM base
    :param model_name: The model name. The model's weights will be saved at root / model_name.
    :param pretrained_val_ratio: The ratio of data to use for validation (for early stopping)
    :param batch_size: The batch size to use when pretraining the model
    :param learning_rate: The learning rate to use
    :param max_epochs: The maximum number of epochs to train the model for
    :param patience: The patience to use for early stopping. If the model trains for
        `patience` epochs without improvement on the validation set, training ends
    """

    model = Pretrainer(
        root,
        batch_size,
        learning_rate,
        classifier_vector_size,
        classifier_dropout,
        classifier_base_layers,
        num_classification_layers,
        pretrained_val_ratio,
        model_name,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=patience, verbose=True, mode="min"
    )
    trainer = pl.Trainer(
        default_save_path=Path(model.root),
        max_epochs=max_epochs,
        early_stop_callback=early_stop_callback,
    )
    trainer.fit(model)

    return model.classifier
