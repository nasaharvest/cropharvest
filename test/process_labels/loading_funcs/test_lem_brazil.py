from process_labels import datasets
from process_labels.loading_funcs.brazil import LABEL_TO_CLASSIFICATION


def test_lem_brazil():

    lem_brazil = datasets.load("lem-brazil")

    non_crop_labels = [key for key, val in LABEL_TO_CLASSIFICATION.items() if val == "non_crop"]

    non_crops = lem_brazil[lem_brazil.label.isin(non_crop_labels)]
    assert (non_crops.is_crop == 0).all()

    crops = lem_brazil[~lem_brazil.label.isin(non_crop_labels)]
    assert (crops.is_crop == 1).all()
