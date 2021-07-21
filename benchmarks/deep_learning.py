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
    DL_PRETRAINED,
    DL_RANDOM,
    DL_MAML,
)

from dl import Classifier, train, pretrain_model, train_maml_model

from typing import Dict, Optional


def run(
    data_folder: Path = DATAFOLDER_PATH,
    state_dict: Optional[Dict] = None,
    model_name: str = DL_RANDOM,
) -> None:
    if model_name != DL_RANDOM:
        assert state_dict is not None

    evaluation_datasets = CropHarvest.create_benchmark_datasets(data_folder)
    results_folder = data_folder / model_name
    results_folder.mkdir(exist_ok=True)

    for dataset in evaluation_datasets:

        sample_sizes = DATASET_TO_SIZES[dataset.id]

        for seed in SHUFFLE_SEEDS:
            dataset.shuffle(seed)
            for sample_size in sample_sizes:
                print(f"Running {model_name} for {dataset}, seed: {seed} with size {sample_size}")

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
                if state_dict is not None:
                    model.load_state_dict(state_dict)

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

    data_folder = DATAFOLDER_PATH

    # we start by making the state_dicts necessary for the pretrained models

    train_maml_model(
        data_folder,
        classifier_base_layers=CLASSIFIER_BASE_LAYERS,
        classifier_dropout=CLASSIFIER_DROPOUT,
        classifier_vector_size=HIDDEN_VECTOR_SIZE,
        num_classification_layers=NUM_CLASSIFICATION_LAYERS,
        model_name=DL_PRETRAINED,
    )

    pretrain_model(
        data_folder,
        classifier_base_layers=CLASSIFIER_BASE_LAYERS,
        classifier_dropout=CLASSIFIER_DROPOUT,
        classifier_vector_size=HIDDEN_VECTOR_SIZE,
        num_classification_layers=NUM_CLASSIFICATION_LAYERS,
        model_name=DL_PRETRAINED,
    )

    for model in [DL_PRETRAINED, DL_MAML, DL_RANDOM]:
        if model != DL_RANDOM:
            state_dict = torch.load(data_folder / model / "state_dict.pth")
        else:
            state_dict = None

        run(data_folder, state_dict, model)
