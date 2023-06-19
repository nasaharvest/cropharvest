import pandas as pd
import geopandas
import numpy as np
from datetime import datetime

from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH
from cropharvest.columns import RequiredColumns, NullableColumns
from ..utils import DATASET_PATH
from .utils import LATLON_CRS

from typing import Dict, Tuple


# isle de france (the smallest region) has
# 100,000 labels - more than all other datasets
# put together. This helps to limit the total dataset
# size and makes the export more feasible
MAX_ROWS_PER_LABEL = 100

GROUP_TO_CLASSIFICATION = {
    "Blé tendre": "cereals",
    "Maïs grain et ensilage": "cereals",
    "Orge": "cereals",
    "Autres céréales": "cereals",
    "Colza": "oilseeds",
    "Tournesol": "oilseeds",
    "Autres oléagineux": "oilseeds",
    "Protéagineux": "leguminous",
    "Plantes à fibres": "other",
    "Gel (surfaces gelées sans production)": "non_crop",
    "Riz": "cereals",
    "Légumineuses à grains": "leguminous",
    "Fourrage": "other",  # fodder
    "Estives et landes": "non_crop",  # pasture
    "Prairies permanentes": "non_crop",
    "Prairies temporaires": "non_crop",
    "Vergers": "fruits_nuts",  # coffee is here even though it should be beverage_spice
    "Vignes": "fruits_nuts",
    "Fruits à coque": "fruits_nuts",
    "Oliviers": "oilseeds",
    "Légumes ou fleurs": "vegetables_melons",
    "Autres cultures industrielles": "beverage_spice",
    "Canne à sucre": "sugar",
}

EXCEPTIONS_TO_CLASSIFICATION = {
    # in Vergers
    "Café / Cacao": "beverage_spice",
    # all in Légumes ou fleurs
    "Bleuet": "other",  # flower
    "Bugle rampante": "other",  # flower
    "Cornille": "other",  # flower
    "Culture sous serre hors sol": "non_crop",
    "Dolique": "other",  # flower
    "Fraise": "fruits_nuts",
    "Géranium": "other",  # flower
    "Horticulture ornementale de plein champ": "other",  # flower
    "Horticulture ornementale sous abri": "non_crop",
    "Légume sous abri": "non_crop",
    "Marguerite": "other",  # flower
    "Navet": "other",  # flower
    "Panais": "root_tuber",
    "Pâquerette": "other",  # flower
    "Primevère": "other",  # flower
    "Petits pois": "leguminous",
    "Pensée": "other",  # flower
    "Pomme de terre de consommation": "root_tuber",
    "Pomme de terre féculière": "root_tuber",
    "Poivron / Piment": "beverage_spice",
    "Radis": "root_tuber",
    "Salsifis": "root_tuber",
    "Topinambour": "other",  # animal feed
    "Véronique": "other",  # flower
}


def load_codification_mapping() -> Dict[str, Tuple[str, str]]:
    df = pd.read_csv(
        DATASET_PATH / "france/Codification_cultures_principales.csv",
        encoding="ISO-8859-1",
        sep=";",
    )
    df = df[df["Libellé Groupe Culture"] != "Divers"]

    mapper: Dict[str, Tuple[str, str]] = {}

    for _, row in df.iterrows():
        if row["Libellé Culture"] in EXCEPTIONS_TO_CLASSIFICATION:
            mapper[row["Code Culture"]] = (
                row["Libellé Culture"],
                EXCEPTIONS_TO_CLASSIFICATION[row["Libellé Culture"]],
            )
        else:
            mapper[row["Code Culture"]] = (
                row["Libellé Culture"],
                GROUP_TO_CLASSIFICATION[row["Libellé Groupe Culture"]],
            )
    return mapper


def _process_france_2019_rpg(df: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    df = df.to_crs(LATLON_CRS)
    mapper = load_codification_mapping()

    # removes the Divers class
    mapped_vals = list(mapper.keys())
    df = df[df.CODE_CULTU.isin(mapped_vals)]

    all_dfs = []
    for unique_code in df.CODE_CULTU.unique():
        code_df = df[df.CODE_CULTU == unique_code][:MAX_ROWS_PER_LABEL]
        if len(code_df) > 0:
            label, classification_label = mapper[unique_code]
            code_df[NullableColumns.LABEL] = label
            code_df[NullableColumns.CLASSIFICATION_LABEL] = classification_label
            all_dfs.append(code_df)

    df = pd.concat(all_dfs)
    df[RequiredColumns.LON] = df.geometry.centroid.x.values
    df[RequiredColumns.LAT] = df.geometry.centroid.y.values
    df[RequiredColumns.COLLECTION_DATE] = datetime(2019, 1, 1)
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2020, EXPORT_END_MONTH, EXPORT_END_DAY)
    df[RequiredColumns.IS_CROP] = np.where(
        (df[NullableColumns.CLASSIFICATION_LABEL] == "non_crop"), 0, 1
    )
    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index
    return df


def load_ile_de_france():
    df = geopandas.read_file(
        DATASET_PATH
        / (
            "france/RPG_2-0_SHP_LAMB93_R11-2019/RPG/1_DONNEES_LIVRAISON_2019/"
            "RPG_2-0_SHP_LAMB93_R11-2019/PARCELLES_GRAPHIQUES.shp"
        )
    )

    return _process_france_2019_rpg(df)


def load_reunion():
    df = geopandas.read_file(
        DATASET_PATH
        / (
            "france/RPG_2-0_SHP_RGR92UTM40S_D974-2019/RPG/1_DONNEES_LIVRAISON_2019/"
            "RPG_2-0_SHP_RGR92UTM40S_D974-2019/PARCELLES_GRAPHIQUES.shp"
        )
    )
    return _process_france_2019_rpg(df)


def load_martinique():
    df = geopandas.read_file(
        DATASET_PATH
        / (
            "france/RPG_2-0_SHP_UTM20W84MART_D972-2019/RPG/1_DONNEES_LIVRAISON_2019/"
            "RPG_2-0_SHP_UTM20W84MART_D972-2019/PARCELLES_GRAPHIQUES.shp"
        )
    )
    return _process_france_2019_rpg(df)
