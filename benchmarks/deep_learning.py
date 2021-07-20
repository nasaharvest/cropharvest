from dl import Classifier, train

import torch

from pathlib import Path
import json

from cropharvest.datasets import CropHarvest
from cropharvest.utils import DATAFOLDER_PATH
from cropharvest.engineer import TestInstance

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
                print(f"Running Random Forest for {dataset}, seed: {seed} with size {sample_size}")

                json_suffix = f"{dataset.id}_{sample_size}_{seed}.json"
                nc_suffix = f"{dataset.id}_{sample_size}_{seed}.nc"

                # train a model
                model = Classifier(
                    input_size=dataset.num_bands,
                    classifier_base_layers=CLASSIFIER_BASE_LAYERS,
                    num_classification_layers=NUM_CLASSIFICATION_LAYERS,
                    classifier_dropout=CLASSIFIER_DROPOUT,
                    classifier_vector_size=HIDDEN_VECTOR_SIZE,
                )

                model = train(model, dataset, sample_size)

                for test_id, test_instance in dataset.test_data():

                    results_json = results_folder / f"{test_id}_{json_suffix}"
                    results_nc = results_folder / f"{test_id}_{nc_suffix}"

                    if results_json.exists():
                        print(f"Results already saved for {results_json} - skipping")

                    test_x = torch.from_numpy(test_instance.x).float()
                    with torch.no_grad():
                        preds = model(test_x).squeeze(dim=1).numpy()
                    results = test_instance.evaluate_predictions(preds)

                    with Path(results_json).open("w") as f:
                        json.dump(results, f)

                    ds = test_instance.to_xarray(preds)
                    ds.to_netcdf(results_nc)
                # finally, we want to get results when all the test instances are considered
                # together
                all_nc_files = list(results_folder.glob(f"*_{nc_suffix}"))
                combined_instance, combined_preds = TestInstance.load_from_nc(all_nc_files)

                combined_results = combined_instance.evaluate_predictions(combined_preds)

                with (results_folder / f"combined_{json_suffix}").open("w") as f:
                    json.dump(combined_results, f)


if __name__ == "__main__":
    run()
