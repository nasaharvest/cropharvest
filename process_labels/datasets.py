import geopandas
import pandas as pd
from pathlib import Path
from . import loading_funcs

from cropharvest.utils import DATAFOLDER_PATH
from cropharvest.config import LABELS_FILENAME
from .utils import add_is_test_column

from typing import cast, Callable, List, Optional

DATASETS = {
    "ethiopia": {
        "function": loading_funcs.load_ethiopia,
        "description": "Hand-labelled crop / non-crop labels in Ethiopia",
    },
    "sudan": {
        "function": loading_funcs.load_sudan,
        "description": "Hand-labelled crop / non crop labels in Sudan",
    },
    "togo": {
        "function": loading_funcs.load_togo,
        "description": "Hand-labelled crop / non crop labels in Togo",
    },
    "togo-eval": {
        "function": loading_funcs.load_togo_eval,
        "description": (
            "Hand-labelled crop / non crop labels in Togo. "
            "These labels are a consensus set collected from 4 labellers."
        ),
    },
    "lem-brazil": {
        "function": loading_funcs.load_lem_brazil,
        "description": (
            "Open source land cover labels collected in Bahia, Brazil. "
            "For more information, please refer to "
            "https://www.sciencedirect.com/science/article/pii/S2352340920314359"
        ),
    },
    "geowiki-landcover-2017": {
        "function": loading_funcs.load_geowiki_landcover_2017,
        "description": (
            "Open source crop / non crop labels collected globally using "
            "GeoWiki. For more information, please refer to "
            "https://doi.pangaea.de/10.1594/PANGAEA.873912"
        ),
    },
    "central-asia": {
        "function": loading_funcs.load_central_asia,
        "description": (
            "Open source crop type labels collected in central asia. "
            "For more information, please refer to "
            "https://www.nature.com/articles/s41597-020-00591-2.pdf"
        ),
    },
    "rwanda": {
        "function": loading_funcs.load_rwanda,
        "description": (
            "Open source crop type labels in Rwanda. For more "
            "information, please refer to "
            "https://doi.org/10.34911/rdnt.r4p1fr"
        ),
    },
    "kenya": {
        "function": loading_funcs.load_kenya,
        "description": (
            "Open source crop type labels in Kenya. For more "
            "information, please refer to "
            "https://doi.org/10.34911/rdnt.u41j87"
        ),
    },
    "kenya-non-crop": {
        "function": loading_funcs.load_kenya_non_crop,
        "description": "Hand-labelled non crop labels in Kenya",
    },
    "uganda": {
        "function": loading_funcs.load_uganda,
        "description": (
            "Open source crop type labels in Uganda. For more "
            "information, please refer to "
            "https://registry.mlhub.earth/10.34911/rdnt.eii04x/"
        ),
    },
    "tanzania": {
        "function": loading_funcs.load_tanzania,
        "description": (
            "Open source crop type labels in Tanzania, For "
            "more information, please refer to "
            "https://doi.org/10.34911/rdnt.5vx40r"
        ),
    },
    "usa-kern": {
        "function": loading_funcs.load_kern_2020,
        "description": (
            "Open source crop type labels submitted by farmers "
            "in Kern, California for 2020. For more information "
            "please refer to "
            "http://www.kernag.com/gis/gis-data.asp#datumused"
        ),
    },
    "croplands": {
        "function": loading_funcs.load_croplands,
        "description": (
            "Open source crop / non crop and crop type labels "
            "with global coverage collected by the GFSAD "
            "project (https://croplands.org/home) retrieved from "
            "https://croplands.org/app/data/search?page=1&page_size=200 "
        ),
    },
    "zimbabwe": {
        "function": loading_funcs.load_zimbabwe,
        "description": "Maize labels collected by the FEWS NET",
    },
    "mali": {
        "function": loading_funcs.load_mali,
        "description": (
            "Crop type labels collected in Segou, Mali for 2019 and 2018 "
            "collected as part of the Relief to Resistance in the Sahel "
            "(R2R)"
        ),
    },
    "mali-non-crop": {
        "function": loading_funcs.load_mali_crop_noncrop,
        "description": "Hand labelled non-crop labels in Mali",
    },
    "ile-de-france": {
        "function": loading_funcs.load_ile_de_france,
        "description": (
            "2019 data from France's Registre parcellaire graphique (RPG) "
            "in the Ile de France region. Retrieved from "
            "ftp://RPG_ext:quoojaicaiqu6ahD@ftp3.ign.fr/RPG_2-0__SHP_LAMB93_R11-2019_2019-01-15.7z"
            "on May 4th 2021. When loaded from the raw data, the dataset size is significantly "
            "reduced (i.e. we take a small subset of the total available labels) "
        ),
    },
    "brazil-non-crop": {
        "function": loading_funcs.load_brazil_noncrop,
        "description": {"Hand labelled non-crop labels in Brazil"},
    },
    "reunion-france": {
        "function": loading_funcs.load_reunion,
        "description": (
            "2019 data from France's Registre parcellaire graphique (RPG) "
            "in Réunion. Retrieved from "
            "ftp://RPG_ext:quoojaicaiqu6ahD@ftp3.ign.fr/"
            "RPG_2-0__SHP_RGR92UTM40S_D974-2019_2019-01-15.7z"
            "on June 2nd 2021. When loaded from the raw data, the dataset size is significantly "
            "reduced (i.e. we take a small subset of the total available labels) "
        ),
    },
    "martinique-france": {
        "function": loading_funcs.load_martinique,
        "description": (
            "2019 data from France's Registre parcellaire graphique (RPG) "
            "in Martinique. Retrieved from "
            "ftp://RPG_ext:quoojaicaiqu6ahD@ftp3.ign.fr/"
            "RPG_2-0__SHP_UTM20W84MART_D972-2019_2019-01-15.7z"
            "on June 2nd 2021. When loaded from the raw data, the dataset size is significantly "
            "reduced (i.e. we take a small subset of the total available labels) "
        ),
    },
    "rwanda-ceo": {
        "function": loading_funcs.load_rwanda_ceo,
        "description": "Hand-labelled crop / non crop labels in Rwanda",
    },
}


NON_NULLABLE_COLUMNS = [
    "index",
    "is_crop",
    "lat",
    "lon",
    "dataset",
    "collection_date",
    "export_end_date",
    "geometry",
]
NULLABLE_COLUMNS = ["harvest_date", "planting_date", "label", "classification_label"]


def load(dataset_name: str) -> geopandas.GeoDataFrame:
    return cast(Callable, DATASETS[dataset_name]["function"])()


def describe(dataset_name: str) -> str:
    return cast(str, DATASETS[dataset_name]["description"])


def list_datasets() -> List[str]:
    return list(DATASETS.keys())


def combine_datasets(ignore_datasets: Optional[List[str]] = None) -> geopandas.GeoDataFrame:
    all_datasets: List[geopandas.GeoDataFrame] = []
    all_columns = NON_NULLABLE_COLUMNS + NULLABLE_COLUMNS

    for dataset_name in list_datasets():
        if (ignore_datasets is not None) and (dataset_name in ignore_datasets):
            continue
        dataset = load(dataset_name)
        dataset = dataset.assign(dataset=dataset_name)

        for column in NULLABLE_COLUMNS:
            if column not in dataset:
                dataset = dataset.assign(**{column: None if "date" not in column else pd.NaT})
        all_datasets.append(dataset[all_columns])
    dataset = pd.concat(all_datasets)
    # finally, some updates to the labels to make them more homogeneous
    dataset["label"] = dataset.label.str.lower().replace(" ", "_")
    return add_is_test_column(dataset)


def update_processed_datasets(data_folder: Path = DATAFOLDER_PATH) -> None:

    combined_datasets = combine_datasets()
    combined_datasets.to_file(data_folder / LABELS_FILENAME, driver="GeoJSON")
