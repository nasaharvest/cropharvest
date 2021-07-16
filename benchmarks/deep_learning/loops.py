from tqdm import tqdm
from torch.optim import Adam

from cropharvest.datasets import CropHarvest

from .lstm import Classifier
from ..config import PATIENCE

from typing import Optional


def train(
    classifier: Classifier,
    dataset: CropHarvest,
    sample_size: Optional[float],
    num_grad_steps: int = 100,
    learning_rate: float = 0.001,
) -> None:

    opt = Adam(classifier.parameters(), lr=learning_rate)

    for i in tqdm(range(num_grad_steps)):
        classifier.train()
