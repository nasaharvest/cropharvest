from tqdm import tqdm
from torch.optim import Adam
import torch
from torch import nn

from cropharvest.utils import sample_with_memory
from cropharvest.datasets import CropHarvest

from .lstm import Classifier

from typing import Optional, List


def train(
    classifier: Classifier,
    dataset: CropHarvest,
    sample_size: Optional[int],
    num_grad_steps: int = 100,
    learning_rate: float = 0.001,
    k: int = 10,
) -> Classifier:

    opt = Adam(classifier.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss(reduction="mean")

    if sample_size is not None:
        train_batch_total = dataset.sample(sample_size // 2)
        state: List[int] = []

    for i in tqdm(range(num_grad_steps)):
        if i != 0:
            classifier.train()
            opt.zero_grad()

        if sample_size is not None:
            assert train_batch_total is not None
            indices, state = sample_with_memory(
                list(range(train_batch_total[0].shape[0])), k * 2, state
            )
            train_x, train_y = train_batch_total[0][indices], train_batch_total[1][indices]
        else:
            train_x, train_y = dataset.sample(k, deterministic=False)

        preds = classifier(torch.from_numpy(train_x).float()).squeeze(dim=1)
        loss = loss_fn(preds, torch.from_numpy(train_y).float())

        loss.backward()
        opt.step()
    return classifier
