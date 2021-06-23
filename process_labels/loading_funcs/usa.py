import geopandas
import numpy as np
from datetime import datetime
from tqdm import tqdm

from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH

from .utils import LATLON_CRS
from ..columns import NullableColumns, RequiredColumns
from ..utils import DATASET_PATH

from typing import Set


KERN_LABEL_TO_CLASSIFICATION = {
    "FRUIT": "fruits_nuts",  # mostly grapes
    "NUTS": "fruits_nuts",
    "CITRUS": "fruits_nuts",
    "MISC": "other",
    "VEGETABLE": "vegetables_melons",
    "FRUIT_POME": "fruits_nuts",
    "BERRIES": "fruits_nuts",
    "MELONS": "vegetables_melons",
    "FRUIT_TROP": "fruits_nuts",
}

KERN_NOT_VEGETABLE_CLASSIFICATION = {
    "POTATO": "root_tuber",
    "PEPPER FRUITNG": "beverage_spice",
    "FENNEL": "beverage_spice",
    "PARSLEY": "beverage_spice",
    "RUTABAGA": "root_tuber",
    "PEPPER SPICE": "beverage_spice",
    "DILL": "beverage_spice",
    "SWEET POTATO": "root_tuber",
    "CHIVE": "beverage_spice",
    "RADISH": "root_tuber",
    "EGGPLANT": "leguminous",
    "PEAS": "leguminous",
    "BEAN DRIED": "leguminous",
    "DAIKON": "root_tuber",
    "DANDELION GREEN": "other",
    "SWEET BASIL": "beverage_spice",
    "CILANTRO": "beverage_spice",
    "MUSTARD GREENS": "oilseeds",
    "KOHLRABI": "root_tuber",
    "HERB, SPICE": "beverage_spice",
    "PARSNIP": "root_tuber",
    "BEAN DRIED SEED": "leguminous",
    "CORN, SWEET": "cereals",
    "MINT": "beverage_spice",
    "PEPPER FRUIT SD": "beverage_spice",
}

KERN_FRUIT_TREE_TO_CLASSIFICATION = {"OLIVE": "oilseeds", "AVOCADO": "fruits_nuts"}

KERN_FIELD_TO_CLASSIFICATION = {
    "COTTON": "other",
    "ALFALFA": "leguminous",
    "TRITICALE": "cereals",
    "WHEAT": "cereals",
    "OAT": "cereals",
    "SUDANGRASS": "cereals",
    "PASTURELAND": "non_crop",
    "SAFFLOWER": "oilseeds",
    "BARLEY": "cereals",
    "MUSTARD": "oilseeds",
    "SOYBEAN": "oilseeds",
    "SOD FARM (TURF)": "other",
    "GARBANZO BEAN": "leguminous",
    "SORGHUM MILO": "cereals",
    "BERMUDA GRASS": "other",
    "RAPE": "oilseeds",
    "RANGELAND": "non_crop",
    "LOVEGRASS (FORA": "non_crop",
    "FAVA BEAN": "leguminous",
    "SORGHUM": "cereals",
    "CORN, GRAIN": "cereals",
    "CORN": "cereals",
    "FORAGE HAY/SLGE": "other",
    "RYE": "cereals",
    "TURF/SOD": "other",
    "RYEGRAS": "other",
}

OVERLAPPING_THRESHOLD = 0.25


def _remove_overlapping(df: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    rows_to_remove: Set[int] = set()

    org_len = len(df)
    df = df[df.geometry.is_valid]
    print(f"Removed {org_len - len(df)} invalid geometries")

    for idx_1, row_1 in tqdm(df.iterrows(), total=len(df)):
        for idx_2, row_2 in df.iloc[idx_1 + 1 :].iterrows():
            if row_1.geometry.intersects(row_2.geometry):
                max_intersection_area = OVERLAPPING_THRESHOLD * min(
                    row_1.geometry.area, row_2.geometry.area
                )
                if row_1.geometry.intersection(row_2.geometry).area >= max_intersection_area:
                    rows_to_remove.add(idx_1)
                    rows_to_remove.add(idx_2)

    cleaned_df = df.drop(rows_to_remove)

    print(f"New df has len {len(cleaned_df)}, from {len(df)}")
    return cleaned_df


def load_kern_2020():

    df = geopandas.read_file(DATASET_PATH / "usa/kern2020")
    df = df.to_crs(LATLON_CRS)

    # we don't need to differentiate between organic and
    # non organic crops
    df["COMM"] = df.apply(lambda x: x.COMM.split("-")[0], axis=1)
    df["COMM"] = df.apply(lambda x: x.COMM.replace("FOR/FOD", "").strip(), axis=1)

    # remove unclear labels
    df = df[df.SYMBOL != "GREENHOUSE"]
    df = df[df.SYMBOL != "OUTDOOR"]
    df = df[df.COMM != "UNCULTIVATED AG"]
    df = df[df.COMM != "BEAN SUCCULENT"]

    df = _remove_overlapping(df)

    x = df.geometry.centroid.x.values
    y = df.geometry.centroid.y.values
    df[RequiredColumns.LAT] = y
    df[RequiredColumns.LON] = x

    df = df.rename(columns={"COMM": NullableColumns.LABEL})

    def label_to_classification_label(row: geopandas.GeoSeries) -> str:
        if row.SYMBOL == "VEGETABLE":
            if row[NullableColumns.LABEL] in KERN_NOT_VEGETABLE_CLASSIFICATION.keys():
                return KERN_NOT_VEGETABLE_CLASSIFICATION[row[NullableColumns.LABEL]]
            return KERN_LABEL_TO_CLASSIFICATION["VEGETABLE"]
        elif row.SYMBOL == "FRUIT_TREE":
            return KERN_FRUIT_TREE_TO_CLASSIFICATION[row[NullableColumns.LABEL]]
        elif row.SYMBOL == "FIELD":
            return KERN_FIELD_TO_CLASSIFICATION[NullableColumns.LABEL]
        else:
            return KERN_LABEL_TO_CLASSIFICATION[row.SYMBOL]

    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(label_to_classification_label, axis=1)
    df[RequiredColumns.IS_CROP] = np.where(
        (df[NullableColumns.CLASSIFICATION_LABEL] == "non_crop"), 0, 1
    )

    # collection date reported on the website
    df[RequiredColumns.COLLECTION_DATE] = datetime(2020, 6, 10)
    df[RequiredColumns.EXPORT_END_DATE] = datetime(2021, EXPORT_END_MONTH, EXPORT_END_DAY)

    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index

    return df
