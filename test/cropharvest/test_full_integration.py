from cropharvest.datasets import CropHarvest
from cropharvest.inference import Inference
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "data"
TIF_FILE = Path(__file__).parent / "98-togo_2019-02-06_2020-02-01.tif"


def test_full_integration():
    # This test mirrors all the functionality in the demo notebook
    evaluation_datasets = CropHarvest.create_benchmark_datasets(DATA_DIR)
    assert len(evaluation_datasets) == 3, "There should be 3 evaluation datasets"

    togo_dataset = evaluation_datasets[-1]
    X, y = togo_dataset.as_array(flatten_x=True)
    assert X.shape[0] == 1290
    assert y.shape[0] == 1290
    assert X.shape[1] == 216

    model = RandomForestClassifier(random_state=0)
    model.fit(X, y)

    test_preds, test_instances = [], []
    for _, test_instance in togo_dataset.test_data(flatten_x=True):
        test_preds.append(model.predict_proba(test_instance.x)[:, 1])
        test_instances.append(test_instance)

    metrics = test_instances[0].evaluate_predictions(test_preds[0])
    assert metrics["f1_score"] > 0.73, "f1-score should be greater than 0.73"
    assert metrics["auc_roc"] > 0.88, "AUC-ROC should be greater than 0.88"

    preds = Inference(model=model, normalizing_dict=None).run(TIF_FILE)

    # Check size
    assert preds.dims["lat"] == 17
    assert preds.dims["lon"] == 17

    # Check all predictions between 0 and 1
    assert preds.min() >= 0
    assert preds.max() <= 1
