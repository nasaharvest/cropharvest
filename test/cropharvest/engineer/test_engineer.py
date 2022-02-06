from pathlib import Path
from datetime import datetime
import numpy as np

from cropharvest.engineer import Engineer, BANDS, DYNAMIC_BANDS, STATIC_BANDS
from cropharvest.config import DEFAULT_NUM_TIMESTEPS

from typing import Dict, Union

TIF_FILE = Path(__file__).parent / "98-togo_2019-02-06_2020-02-01.tif"


def test_load_tif_file():

    loaded_file, _ = Engineer.load_tif(TIF_FILE, start_date=datetime(2019, 2, 6))
    assert loaded_file.shape[0] == DEFAULT_NUM_TIMESTEPS
    assert loaded_file.shape[1] == len(DYNAMIC_BANDS) + len(STATIC_BANDS)

    # also, check the static bands are actually constant across time
    static_bands = loaded_file.values[:, len(DYNAMIC_BANDS)]
    for i in range(1, DEFAULT_NUM_TIMESTEPS):
        assert np.array_equal(static_bands[0], static_bands[i], equal_nan=True)

    # finally, check expected for temperature
    temperature_band = DYNAMIC_BANDS.index("temperature_2m")
    temperature_values = loaded_file.values[:, temperature_band, :, :]
    assert ((temperature_values) > 0).all()  # in Kelvin
    # https://en.wikipedia.org/wiki/Highest_temperature_recorded_on_Earth
    assert ((temperature_values) < 329.85).all()


def test_fillna_real():

    loaded_file, average_slope = Engineer.load_tif(TIF_FILE, start_date=datetime(2019, 2, 6))

    # slope is calculated from neighbouring points, so the
    # edges are NaN
    for lat_idx in range(loaded_file.shape[2]):
        for lon_idx in range(loaded_file.shape[3]):
            # we remove the first index to simulate removing B1 and B10
            array = loaded_file.values[:, 2:, lat_idx, lon_idx]
            num_timesteps = array.shape[0]
            # and we add an array to simulate adding NDVI
            array = np.concatenate([array, np.ones([num_timesteps, 1])], axis=1)
            new_array = Engineer.fillna(array, average_slope)
            if np.isnan(array[-2]).all():
                assert (new_array[-2] == average_slope).all()
            assert not np.isnan(new_array).any()


def test_fillna_simulated():

    array = np.array([[1, float("NaN"), 3]] * len(BANDS)).T
    expected_array = np.array([[1, 2, 3]] * len(BANDS)).T
    new_array = Engineer.fillna(array, average_slope=1)
    assert np.array_equal(new_array, expected_array)


def test_normalizing_dict():
    class TestEngineer(Engineer):
        def __init__(self):
            self.norm_interim: Dict[str, Union[np.ndarray, int]] = {"n": 0}

    engineer = TestEngineer()
    array_1 = np.array([[1, 2, 3], [2, 2, 4]])
    array_2 = np.array([[2, 2, 4], [1, 2, 3]])

    engineer.update_normalizing_values(array_1)
    engineer.update_normalizing_values(array_2)

    normalizing_values = engineer.calculate_normalizing_dict()

    assert (normalizing_values["mean"] == np.array([1.5, 2, 3.5])).all()
    assert (
        normalizing_values["std"]
        == np.array([np.std([1, 2, 2, 1], ddof=1), 0, np.std([3, 4, 4, 3], ddof=1)])
    ).all()


def test_find_nearest():
    array = np.array([1, 2, 3, 4, 5])

    target = 1.1

    assert Engineer.find_nearest(array, target) == 1


def test_filename_correctly_processed():

    idx, dataset = Engineer.process_filename(TIF_FILE.name)
    assert idx == 98
    assert dataset == "togo"


def test_filename_correctly_processed_2():

    filename = "98-geowiki-landcover-2017_2019-02-06_2020-02-01.tif"
    idx, dataset = Engineer.process_filename(filename)
    assert idx == 98
    assert dataset == "geowiki-landcover-2017"
