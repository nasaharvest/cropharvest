from pathlib import Path
import json
from copy import deepcopy

from cropharvest.utils import DATAFOLDER_PATH
from cropharvest.datasets import CropHarvest, Task, CropHarvestLabels
from cropharvest.engineer import TestInstance

from sklearn.ensemble import RandomForestClassifier


METHOD = "BoundingBoxes"


def select_points(evaluation_dataset: CropHarvest, all_labels: CropHarvestLabels) -> CropHarvest:
    """
    This is what participants would implement.
    Given an evaluation dataset, they would need
    to implement some method of selecting points against
    which a model will be trained. In this example code, I
    show two examples - one uses bounding boxes, the other directly
    overwrites the CropHarvestLabels geojson
    """
    if METHOD == "BoundingBoxes":
        training_task = deepcopy(evaluation_dataset.task)
        training_task.test_identifier = None
        return CropHarvest(evaluation_dataset.root, training_task)
    elif METHOD == "manual_modification":
        # if we don't set a bounding box for the task, no spatial filtering
        # happens on the labels
        training_task = Task(
            target_label=evaluation_dataset.task.target_label,
            balance_negative_crops=evaluation_dataset.task.balance_negative_crops,
        )

        # let's manually select points in the labels to be within the bounding box.
        # 1. Make a new geojson. We do it according to the bounding boxes but this could be done
        # in any way
        filtered_geojson = all_labels.filter_geojson(
            all_labels.as_geojson(),
            evaluation_dataset.task.bounding_box,
            include_external_contributions=True,
        )
        # 2. make a new CropHarvestLabels object with this new geojson
        new_labels = deepcopy(all_labels)
        new_labels.update(filtered_geojson)
        # create a CropHarvest training task with this data
        training_dataset = CropHarvest(evaluation_dataset.root, training_task)
        training_dataset.update_labels(new_labels)
        return training_dataset


def train_and_eval(
    training_dataset: CropHarvest, evaluation_dataset: CropHarvest, results_folder: Path
):
    train_x, train_y = training_dataset.as_array(flatten_x=True)
    # train a model
    model = RandomForestClassifier()
    model.fit(train_x, train_y)

    json_suffix = f"{training_dataset.id}.json"
    nc_suffix = f"{training_dataset.id}.nc"
    for test_id, test_instance in evaluation_dataset.test_data(flatten_x=True, max_size=10000):
        results_json = results_folder / f"{test_id}_{json_suffix}"
        results_nc = results_folder / f"{test_id}_{nc_suffix}"

        if results_json.exists():
            print(f"Results already saved for {results_json} - skipping")

        preds = model.predict_proba(test_instance.x)[:, 1]

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


def main():
    evaluation_datasets = CropHarvest.create_benchmark_datasets(DATAFOLDER_PATH)
    all_labels = CropHarvestLabels(DATAFOLDER_PATH)
    results_folder = DATAFOLDER_PATH / "data_centric_test"
    results_folder.mkdir(exist_ok=True)

    for evaluation_dataset in evaluation_datasets:
        training_dataset = select_points(evaluation_dataset, all_labels)
        train_and_eval(training_dataset, evaluation_dataset, results_folder)


if __name__ == "__main__":
    main()
