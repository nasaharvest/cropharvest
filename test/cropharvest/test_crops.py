from cropharvest.crops import to_one_hot, CropClassifications

import numpy as np
from typing import List


def test_to_one_hot():

    encodings: List = []
    for x in CropClassifications:
        encodings.append(to_one_hot(x.name))
    encodings_np = np.array(encodings)

    for i in range(encodings_np.shape[0]):
        assert encodings_np[i].sum() == 1
    for i in range(encodings_np.shape[1]):
        assert encodings_np[:, i].sum() == 1

    encodings_crop = to_one_hot("crop")
    assert np.isclose(sum(encodings_crop), 1)

    nonzero = [x for x in encodings_crop if x != 0]
    assert len(nonzero) == (len(CropClassifications) - 1)
