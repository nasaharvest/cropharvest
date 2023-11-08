from pathlib import Path
import pandas as pd
import json

from cropharvest.utils import DATAFOLDER_PATH
from cropharvest.datasets import CropHarvest, Task, CropHarvestLabels
from cropharvest.engineer import TestInstance
from cropharvest.columns import RequiredColumns

from sklearn.ensemble import RandomForestClassifier


def select_points(evaluation_dataset: CropHarvest, all_labels: CropHarvestLabels) -> pd.DataFrame:
    """
    This is what participants would implement.
    Given an evaluation dataset, they would need
    to implement some method of selecting points against
    which a model will be trained. In this example code, I
    show two examples - one uses bounding boxes, the other directly
    overwrites the CropHarvestLabels geojson
    """
    # let's manually select points in the labels to be within the bounding box.
    # 1. Make a new geojson. We do it according to the bounding boxes but this could be done
    # in any way
    filtered_geojson = all_labels.filter_geojson(
        all_labels.as_geojson(),
        evaluation_dataset.task.bounding_box,
        include_external_contributions=True,
    )

    # the csv will contain the ids and datasets of the selected rows
    return pd.DataFrame(filtered_geojson[[RequiredColumns.DATASET, RequiredColumns.INDEX]])


def train_and_eval(
    training_labels: pd.DataFrame, evaluation_dataset: CropHarvest, results_folder: Path
):
    # 1. we make a training dataset from the labels
    labels = CropHarvestLabels(DATAFOLDER_PATH)
    filtered_labels = labels.as_geojson().merge(
        training_labels, on=[RequiredColumns.DATASET, RequiredColumns.INDEX]
    )
    labels._labels = filtered_labels
    training_dataset = CropHarvest(
        evaluation_dataset.root,
        Task(
            target_label=evaluation_dataset.task.target_label,
            balance_negative_crops=evaluation_dataset.task.balance_negative_crops,
        ),
    )
    training_dataset.update_labels(labels)

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

    togo_eval = [x for x in evaluation_datasets if "Togo" in x.task.name]
    training_points_df = select_points(togo_eval, all_labels)
    train_and_eval(training_points_df, togo_eval, results_folder)


if __name__ == "__main__":
    main()
