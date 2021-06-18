import numpy as np

from typing import List


# These crop classifications are pulled from the
# FAO's indicative crop classification:
# https://stats-class.fao.uniroma2.it/caliper/classification-page/43
# A csv containing these classifications is saved in /data/ICC11-core.csv
# When new datasets are added, any crop types (i.e. rows where label is not None)
# should also contain a classification_label column. This is done manually due to
# inconsistencies in label names
CROP_CLASSIFICATIONS = {
    "non_crop": 0,
    "cereals": 1,
    "vegetables_melons": 2,
    "fruits_nuts": 3,
    "oilseeds": 4,
    "root_tuber": 5,
    "beverage_spice": 6,
    "leguminous": 7,
    "sugar": 8,
    "other": 9,
}


def to_one_hot(crop_name: str) -> List[float]:

    if crop_name in CROP_CLASSIFICATIONS:
        encoding = np.zeros(len(CROP_CLASSIFICATIONS))
        encoding[CROP_CLASSIFICATIONS[crop_name]] = 1
    elif crop_name == "crop":
        encoding = np.ones(len(CROP_CLASSIFICATIONS))
        encoding[CROP_CLASSIFICATIONS["non_crop"]] = 0
        # normalize the one hot encoding
        encoding /= encoding.sum()
    else:
        raise RuntimeError(f"Unrecognized crop type {crop_name}")
    return encoding.tolist()
