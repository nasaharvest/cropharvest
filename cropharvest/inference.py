from cropharvest.engineer import Engineer
from cropharvest.utils import TORCH_INSTALLED
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Dict, Optional, Tuple, Union
import numpy as np
import xarray as xr
import pandas as pd
import re

if TORCH_INSTALLED:
    import torch


class Inference:
    """
    Class for running inference on a single tif file using an sklearn or
    pytorch model (including jit).
    """

    def __init__(
        self,
        model,
        device=None,
        batch_size: int = 64,
        normalizing_dict: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.model = model
        self.device = device
        self.normalizing_dict = normalizing_dict
        self.batch_size: int = batch_size

        if hasattr(self.model, "predict_proba"):
            self.model_type = "sklearn"
        elif TORCH_INSTALLED:
            self.model_type = "pytorch"
        else:
            raise ModuleNotFoundError(
                "Using PyTorch model but PyTorch is not installed. Please pip install torch"
            )

    @staticmethod
    def start_date_from_str(path: Union[Path, str]) -> datetime:
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", str(path))
        if len(dates) != 2:
            raise ValueError(f"{path} should have start and end date")
        start_date_str, _ = dates
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        return start_date

    @staticmethod
    def _tif_to_np(
        local_path: Path,
        start_date: datetime,
        normalizing_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        da, slope = Engineer.load_tif(local_path, start_date=start_date)

        # Process remote sensing data
        x_np = da.values
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
        x_np = np.moveaxis(x_np, -1, 0)
        x_np = Engineer.calculate_ndvi(x_np)
        x_np = Engineer.remove_bands(x_np)
        x_np = Engineer.fillna(x_np, slope)
        if normalizing_dict is not None:
            x_np = (x_np - normalizing_dict["mean"]) / normalizing_dict["std"]

        # Get lat lons
        lon, lat = np.meshgrid(da.x.values, da.y.values)
        flat_lat, flat_lon = (
            np.squeeze(lat.reshape(-1, 1), -1),
            np.squeeze(lon.reshape(-1, 1), -1),
        )
        return x_np, flat_lat, flat_lon

    @staticmethod
    def _combine_predictions(
        flat_lat: np.ndarray, flat_lon: np.ndarray, batch_predictions: List[np.ndarray]
    ) -> xr.Dataset:
        all_preds = np.concatenate(batch_predictions, axis=0)
        if len(all_preds.shape) == 1:
            all_preds = np.expand_dims(all_preds, axis=-1)

        data_dict: Dict[str, np.ndarray] = {"lat": flat_lat, "lon": flat_lon}
        for i in range(all_preds.shape[1]):
            prediction_label = f"prediction_{i}"
            data_dict[prediction_label] = all_preds[:, i]
        return pd.DataFrame(data=data_dict).set_index(["lat", "lon"]).to_xarray()

    def _on_single_batch(self, batch_x_np: np.ndarray) -> np.ndarray:
        if self.model_type == "sklearn":
            flattened_batch = batch_x_np.reshape(batch_x_np.shape[0], -1)
            return self.model.predict_proba(flattened_batch)[:, 1]
        elif self.model_type == "pytorch":
            batch_x = torch.from_numpy(batch_x_np).float()
            if self.device is not None:
                batch_x = batch_x.to(self.device)
            with torch.no_grad():
                preds = self.model(batch_x)
            return preds.cpu().numpy()
        else:
            # This code should never be reached
            raise ValueError(f"Unknown model type {self.model_type}")

    def run(
        self,
        local_path: Path,
        start_date: Optional[datetime] = None,
        dest_path: Optional[Path] = None,
    ) -> xr.Dataset:
        if start_date is None:
            start_date = self.start_date_from_str(local_path)
        x_np, flat_lat, flat_lon = self._tif_to_np(local_path, start_date, self.normalizing_dict)
        batches = [
            x_np[i : i + self.batch_size] for i in range(0, (x_np.shape[0] - 1), self.batch_size)
        ]
        batch_predictions = [self._on_single_batch(b) for b in batches]
        combined_pred = self._combine_predictions(flat_lat, flat_lon, batch_predictions)
        if dest_path is not None:
            combined_pred.to_netcdf(dest_path)
        return combined_pred
