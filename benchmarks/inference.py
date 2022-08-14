import torch

from datetime import datetime, timedelta
from pathlib import Path

from cropharvest.utils import DATAFOLDER_PATH
from cropharvest.engineer import TestInstance, Engineer
from cropharvest.config import (
    EXPORT_END_DAY,
    EXPORT_END_MONTH,
    DEFAULT_NUM_TIMESTEPS,
    DAYS_PER_TIMESTEP,
)

from config import (
    CLASSIFIER_DROPOUT,
    NUM_CLASSIFICATION_LAYERS,
    HIDDEN_VECTOR_SIZE,
    CLASSIFIER_BASE_LAYERS,
)

from dl import Classifier

from typing import Dict, Optional, Generator, Tuple, List

MAX_INFERENCE_BATCH_SIZE = 10000


def _test_batch(
    filepath: Path, test_instance: TestInstance, batch_size: int = MAX_INFERENCE_BATCH_SIZE
) -> Generator[Tuple[str, TestInstance], None, None]:
    if len(test_instance) > batch_size:
        cur_idx = 0
        while (cur_idx * batch_size) < len(test_instance):
            sub_array = test_instance[cur_idx * batch_size : (cur_idx + 1) * batch_size]
            test_id = f"{cur_idx}_{filepath.stem}"
            cur_idx += 1
            yield test_id, sub_array


def run(
    tif_path: List[Path],
    savefolder_name: str,
    data_folder: Path = DATAFOLDER_PATH,
    state_dict: Optional[Dict] = None,
) -> None:

    results_folder = data_folder / savefolder_name
    results_folder.mkdir(exist_ok=True)

    # train a model
    model = Classifier(
        input_size=18,
        classifier_base_layers=CLASSIFIER_BASE_LAYERS,
        num_classification_layers=NUM_CLASSIFICATION_LAYERS,
        classifier_dropout=CLASSIFIER_DROPOUT,
        classifier_vector_size=HIDDEN_VECTOR_SIZE,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)

    model.eval()

    start_date = datetime(2020, EXPORT_END_MONTH, EXPORT_END_DAY) - timedelta(
        days=DEFAULT_NUM_TIMESTEPS * DAYS_PER_TIMESTEP
    )

    for idx, path_to_file in enumerate(tif_path):
        final_x, flat_lat, flat_lon = Engineer.process_test_file(
            path_to_file, start_date=start_date
        )
        test_instance = TestInstance(x=final_x, y=None, lats=flat_lon, lons=flat_lon)

        test_x = torch.from_numpy(test_instance.x).float()
        with torch.no_grad():
            preds = model(test_x).squeeze(dim=1).numpy()

        ds = test_instance.to_xarray(preds)
        ds.to_netcdf(results_folder / f"{idx}.nc")
    # finally, we want to get results when all the test instances are considered
    # together
    all_nc_files = list(results_folder.glob(f"*.nc"))
    _, combined_preds = TestInstance.load_from_nc(all_nc_files)
    combined_preds.to_netcdf(results_folder / "combined_results.nc")


if __name__ == "__main__":

    # user defined variables
    tif_path = None  # can't be None!
    assert tif_path is not None
    data_folder = DATAFOLDER_PATH
    checkpoint = True
    state_dict_path: Optional[Path] = None

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path)
    else:
        state_dict = None

    start_time = datetime.now()
    print(start_time)
    run(tif_path, "experimental_results", data_folder, state_dict)
    print(datetime.now() - start_time)
