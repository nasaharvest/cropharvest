from dl import Classifier, train

import torch

from pathlib import Path
import json

import sys

sys.path.append("..")

from cropharvest.datasets import CropHarvest
from cropharvest.utils import DATAFOLDER_PATH

from config import (
    SHUFFLE_SEEDS,
    DATASET_TO_SIZES,
    CLASSIFIER_DROPOUT,
    NUM_CLASSIFICATION_LAYERS,
    HIDDEN_VECTOR_SIZE,
    CLASSIFIER_BASE_LAYERS,
)


def run(data_folder: Path = DATAFOLDER_PATH) -> None:
    evaluation_datasets = CropHarvest.create_benchmark_datasets(data_folder)
    results_folder = data_folder / "DL_RANDOM"
    results_folder.mkdir(exist_ok=True)

    for dataset in evaluation_datasets[-1:]:

        sample_sizes = DATASET_TO_SIZES[dataset.id]

        for seed in SHUFFLE_SEEDS:
            dataset.shuffle(seed)
            for sample_size in sample_sizes:
                print(f"Running Random DL for {dataset}, seed: {seed} with size {sample_size}")

                results_json = results_folder / f"{dataset.id}_{sample_size}_{seed}.json"
                results_nc = results_folder / f"{dataset.id}_{sample_size}_{seed}.nc"
                if results_json.exists():
                    print(f"Results already saved for {results_json} - skipping")

                # train a model
                model = Classifier(
                    input_size=dataset.num_bands,
                    classifier_base_layers=CLASSIFIER_BASE_LAYERS,
                    num_classification_layers=NUM_CLASSIFICATION_LAYERS,
                    classifier_dropout=CLASSIFIER_DROPOUT,
                    classifier_vector_size=HIDDEN_VECTOR_SIZE,
                )

                model = train(model, dataset, sample_size)

                for _, test_instance in dataset.test_data():
                    test_x = torch.from_numpy(test_instance.x).float()
                    with torch.no_grad():
                        preds = model(test_x).squeeze(dim=1).numpy()
                    results = test_instance.evaluate_predictions(preds)

                    with Path(results_json).open("w") as f:
                        json.dump(results, f)

                    ds = test_instance.to_xarray(preds)
                    ds.to_netcdf(results_nc)


if __name__ == "__main__":
    run()
