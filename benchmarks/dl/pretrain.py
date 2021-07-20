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
from cropharvest.utils import deterministic_shuffle

from typing import List


class PreTrainCropHarvest(CropHarvest):
    def __init__(self, root, val_ratio: float, download=False, val: bool = False):
        super().__init__(root, task=None, download=download)

        # retrieve the positive and negative filepaths
        positive_paths: List[Path] = []
        negative_paths: List[Path] = []
        for idx, filepath in enumerate(self.filepaths):
            if self.y_vals[idx] == 1:
                positive_paths.append(filepath)
            else:
                negative_paths.append(filepath)

        # the fixed seed is to ensure the validation set is always
        # different from the training set
        positive_paths = deterministic_shuffle(positive_paths, seed=42)
        negative_paths = deterministic_shuffle(negative_paths, seed=42)

        if val:
            positive_paths = positive_paths[: int(len(positive_paths) * val_ratio)]
            negative_paths = negative_paths[: int(len(negative_paths) * val_ratio)]
        else:
            positive_paths = positive_paths[int(len(positive_paths) * val_ratio) :]
            negative_paths = negative_paths[int(len(negative_paths) * val_ratio) :]

        self.filepaths: List[Path] = positive_paths + negative_paths
        self.y_vals: List[int] = [1] * len(positive_paths) + [0] * len(negative_paths)
        self.positive_indices = list(range(len(positive_paths)))
        self.negative_indices = list(
            range(len(positive_paths), len(positive_paths) + len(negative_paths))
        )


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

    def get_dataset(self, is_val: bool) -> PreTrainCropHarvest:
        return PreTrainCropHarvest(
            self.root, download=True, val=is_val, val_ratio=self.pretrained_val_ratio
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
    pretrained_val_ratio: float,
    model_name: str,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    max_epochs: int = 1000,
    patience: int = 10,
) -> pl.LightningModule:

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

    return model
