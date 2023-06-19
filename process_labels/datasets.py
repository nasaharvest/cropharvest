import geopandas
import pandas as pd
from pathlib import Path

from . import loading_funcs
from .utils import add_is_test_column

from cropharvest.columns import NullableColumns, RequiredColumns
from cropharvest.utils import DATAFOLDER_PATH
from cropharvest.config import LABELS_FILENAME

from typing import cast, Callable, List, Optional

DATASETS = {
    "ethiopia": {
        "function": loading_funcs.load_ethiopia,
        "description": "Hand-labelled crop / non-crop labels in Ethiopia",
        "externally_contributed": False,
    },
    "sudan": {
        "function": loading_funcs.load_sudan,
        "description": "Hand-labelled crop / non crop labels in Sudan",
        "externally_contributed": False,
    },
    "togo": {
        "function": loading_funcs.load_togo,
        "description": "Hand-labelled crop / non crop labels in Togo",
        "externally_contributed": False,
    },
    "togo-eval": {
        "function": loading_funcs.load_togo_eval,
        "description": (
            "Hand-labelled crop / non crop labels in Togo. "
            "These labels are a consensus set collected from 4 labellers."
        ),
        "externally_contributed": False,
    },
    "lem-brazil": {
        "function": loading_funcs.load_lem_brazil,
        "description": (
            "Open source land cover labels collected in Bahia, Brazil. "
            "For more information, please refer to "
            "https://www.sciencedirect.com/science/article/pii/S2352340920314359"
        ),
        "externally_contributed": False,
    },
    "geowiki-landcover-2017": {
        "function": loading_funcs.load_geowiki_landcover_2017,
        "description": (
            "Open source crop / non crop labels collected globally using "
            "GeoWiki. For more information, please refer to "
            "https://doi.pangaea.de/10.1594/PANGAEA.873912"
        ),
        "externally_contributed": False,
    },
    "central-asia": {
        "function": loading_funcs.load_central_asia,
        "description": (
            "Open source crop type labels collected in central asia. "
            "For more information, please refer to "
            "https://www.nature.com/articles/s41597-020-00591-2.pdf"
        ),
        "externally_contributed": False,
    },
    "kenya": {
        "function": loading_funcs.load_kenya,
        "description": (
            "Open source crop type labels in Kenya. For more "
            "information, please refer to "
            "https://doi.org/10.34911/rdnt.u41j87"
        ),
        "externally_contributed": False,
    },
    "kenya-non-crop": {
        "function": loading_funcs.load_kenya_non_crop,
        "description": "Hand-labelled non crop labels in Kenya",
        "externally_contributed": False,
    },
    "uganda": {
        "function": loading_funcs.load_uganda,
        "description": (
            "Open source crop type labels in Uganda. For more "
            "information, please refer to "
            "https://registry.mlhub.earth/10.34911/rdnt.eii04x/"
        ),
        "externally_contributed": False,
    },
    "tanzania": {
        "function": loading_funcs.load_tanzania,
        "description": (
            "Open source crop type labels in Tanzania, For "
            "more information, please refer to "
            "https://doi.org/10.34911/rdnt.5vx40r"
        ),
        "externally_contributed": False,
    },
    "croplands": {
        "function": loading_funcs.load_croplands,
        "description": (
            "Open source crop / non crop and crop type labels "
            "with global coverage collected by the GFSAD "
            "project (https://croplands.org/home) retrieved from "
            "https://croplands.org/app/data/search?page=1&page_size=200 "
        ),
        "externally_contributed": False,
    },
    "zimbabwe": {
        "function": loading_funcs.load_zimbabwe,
        "description": "Maize labels collected by the FEWS NET",
        "externally_contributed": False,
    },
    "mali": {
        "function": loading_funcs.load_mali,
        "description": (
            "Crop type labels collected in Segou, Mali for 2019 and 2018 "
            "collected as part of the Relief to Resistance in the Sahel "
            "(R2R)"
        ),
        "externally_contributed": False,
    },
    "mali-non-crop": {
        "function": loading_funcs.load_mali_crop_noncrop,
        "description": "Hand labelled non-crop labels in Mali",
        "externally_contributed": False,
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
        "externally_contributed": False,
    },
    "brazil-non-crop": {
        "function": loading_funcs.load_brazil_noncrop,
        "description": {"Hand labelled non-crop labels in Brazil"},
        "externally_contributed": False,
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
        "externally_contributed": False,
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
        "externally_contributed": False,
    },
    "rwanda-ceo": {
        "function": loading_funcs.load_rwanda_ceo,
        "description": "Hand-labelled crop / non crop labels in Rwanda",
        "externally_contributed": False,
    },
    "canada": {
        "function": loading_funcs.load_canada,
        "description": (
            "Annual Crop Inventory Ground Truth Data from Canada. "
            "For more information, please visit "
            "https://open.canada.ca/data/en/dataset/503a3113-e435-49f4-850c-d70056788632. "
            "Contains information licensed under the Open Government Licence – Canada."
        ),
        "externally_contributed": False,
    },
    "germany": {
        "function": loading_funcs.load_germany,
        "description": (
            "2018 data collected as part of the Common Agricultural Policy "
            " of the European Union, and processed in "
            "https://github.com/lukaskondmann/DENETHOR"
        ),
        "externally_contributed": False,
    },
    "mali-helmets-labelling-crops": {
        "function": loading_funcs.load_mali_hlc,
        "description": ("2022 data collected as part of the Helmets Labelling Crops project"),
        "externally_contributed": False,
    },
    "tanzania-rice-ecaas": {
        "function": loading_funcs.load_tanzania_ecaas,
        "description": "Tanzania Rice ECAAS campaign",
        "externally_contributed": False,
    },
    "tanzania-ceo": {
        "function": loading_funcs.load_tanzania_ceo,
        "description": "Hand-labelled crop / non crop labels in Tanzania",
        "externally_contributed": False,
    },
    "tanania-copernicusgeoglam": {
        "function": loading_funcs.load_tanzania_copernicusgeoglam,
        "description": (
            "Crop Mapping for GEOGLAM Country Level Support"
            "Tanzania Copernicus4Geoglam field campaign"
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/Copernicus4GEOGLAM/"
        ),
        "externally_contributed": False,
    },
    "kenya-coperpenicusgeoglam-shortrain": {
        "function": loading_funcs.load_kenya_copernicusgeoglam_shortrain,
        "description": (
            "Crop Mapping for GEOGLAM Country Level Support"
            "Kenya Copernicus4Geoglam field campaign"
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/Copernicus4GEOGLAM/"
        ),
        "externally_contributed": False,
    },
    "kenya-coperpenicusgeoglam-longrain": {
        "function": loading_funcs.load_kenya_copernicusgeoglam_longrain,
        "description": (
            "Crop Mapping for GEOGLAM Country Level Support"
            "Kenya Copernicus4Geoglam field campaign"
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/Copernicus4GEOGLAM/"
        ),
        "externally_contributed": False,
    },
    "uganda-coperpenicusgeoglam-shortrain": {
        "function": loading_funcs.load_uganda_copernicusgeoglam_shortrain,
        "description": (
            "Crop Mapping for GEOGLAM Country Level Support"
            "Uganda Copernicus4Geoglam field campaign"
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/Copernicus4GEOGLAM/"
        ),
        "externally_contributed": False,
    },
    "uganda-coperpenicusgeoglam-longrain": {
        "function": loading_funcs.load_uganda_copernicusgeoglam_longrain,
        "description": (
            "Crop Mapping for GEOGLAM Country Level Support"
            "Uganda Copernicus4Geoglam field campaign"
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/Copernicus4GEOGLAM/"
        ),
        "externally_contributed": False,
    },
}


def load(dataset_name: str) -> geopandas.GeoDataFrame:
    return cast(Callable, DATASETS[dataset_name]["function"])()


def describe(dataset_name: str) -> str:
    return cast(str, DATASETS[dataset_name]["description"])


def list_datasets() -> List[str]:
    return list(DATASETS.keys())


def combine_datasets(datasets: Optional[List[str]] = None) -> geopandas.GeoDataFrame:
    all_datasets: List[geopandas.GeoDataFrame] = []
    all_columns = NullableColumns.tolist() + RequiredColumns.tolist()

    # the IS_TEST column is the last one to get added, on the combined data
    all_columns.remove(RequiredColumns.IS_TEST)

    if datasets is None:
        datasets = list_datasets()

    for dataset_name in datasets:
        dataset = load(dataset_name)
        dataset = dataset.assign(
            **{
                RequiredColumns.DATASET: dataset_name,
                RequiredColumns.EXTERNALLY_CONTRIBUTED_DATASET: DATASETS[dataset_name][
                    "externally_contributed"
                ],
            }
        )
        for column in NullableColumns.tolist():
            if column not in dataset:
                dataset = dataset.assign(
                    **{column: None if column not in NullableColumns.date_columns() else pd.NaT}
                )
        all_datasets.append(dataset[all_columns])
    dataset = pd.concat(all_datasets)
    # finally, some updates to the labels to make them more homogeneous
    dataset[NullableColumns.LABEL] = dataset.label.str.lower().replace(" ", "_")
    return add_is_test_column(dataset)


def update_processed_datasets(
    data_folder: Path = DATAFOLDER_PATH, overwrite: bool = False
) -> None:
    original_labels: Optional[geopandas.GeoDataFrame] = None
    datasets_to_combine = list_datasets()

    if (not overwrite) and (data_folder / LABELS_FILENAME).exists():
        original_labels = geopandas.read_file(data_folder / LABELS_FILENAME)
        existing_datasets = original_labels[RequiredColumns.DATASET].unique().tolist()
        datasets_to_combine = [x for x in datasets_to_combine if x not in existing_datasets]

    combined_labels = combine_datasets(datasets=datasets_to_combine)

    if original_labels is not None:
        date_columns = RequiredColumns.date_columns() + NullableColumns.date_columns()
        for col in date_columns:
            combined_labels[col] = combined_labels[col].dt.strftime("%Y-%m-%d")
        combined_labels = pd.concat([original_labels, combined_labels])
    combined_labels.to_file(data_folder / LABELS_FILENAME, driver="GeoJSON")
