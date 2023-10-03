import geopandas
import numpy as np
import pandas as pd
from datetime import date, datetime

from cropharvest.columns import RequiredColumns, NullableColumns
from cropharvest.config import EXPORT_END_DAY, EXPORT_END_MONTH

from ..utils import DATASET_PATH
from .utils import LATLON_CRS


LABEL_TO_CLASSIFICATION = {
    "Millet": "cereals",
    "Sorghum": "cereals",
    "Old fallow": "non_crop",
    "Maize": "cereals",
    "Pea": "leguminous",
    "Sesame": "oilseeds",
    "Cowpea": "leguminous",
    "Groundnut": "oilseeds",
    "Fallow": "non_crop",
    "Soybean": "oilseeds",
    "Cotton": "other",
    "Young fallow": "non_crop",
    "Rice": "cereals",
    "Eucalyptus": "other",
    "Cashew tree": "fruits_nuts",
    "Cassava": "root_tuber",
    "Tomato": "vegetables_melons",
    "Potato": "root_tuber",
    "Grasses and other fodder crop": "other",
    "Pine": "fruits_nuts",
    "Fruit crop": "fruits_nuts",
    "Sweet potato": "root_tuber",
    "Cash woody crop": "other",
    "Bean": "leguminous",
    "Apple tree": "fruits_nuts",
    "Cucumber": "vegetables_melons",
    "Taro": "root_tuber",
    "Cabbage": "vegetables_melons",
    "Barley": "cereals",
    "Carrot": "vegetables_melons",
    "Leafy or stem vegetable": "vegetables_melons",
    "Pineapple": "fruits_nuts",
    "Oilseed crop": "oilseeds",
    "Onion": "vegetables_melons",
    "Mango tree": "fruits_nuts",
    "Pear tree": "fruits_nuts",
    "Root, bulb or tuberous vegetable": "root_tuber",
    "Fruit-bearing vegetable": "fruits_nuts",
    "Agricultural bare soil": "non_crop",
    "Asparagus": "vegetables_melons",
    "Vineyard": "fruits_nuts",
    "Market gardening": "other",
    "Peach tree": "fruits_nuts",
    "Leguminous": "leguminous",
    "Cereals": "cereals",
    "Coffee": "beverage_spice",
    "Oat": "cereals",
    "Watermelon": "fruits_nuts",
    "Hibiscus": "beverage_spice",
    "Gombo": "vegetables_melons",
    "Citrus tree": "fruits_nuts",
    "Cauliflower and brocoli": "vegetables_melons",
    "Root/tuber crop with high starch or inulin content": "root_tuber",
    "Eggplant": "vegetables_melons",
    "Mixed annual crops": "other",
    "Weakly vegetated agricultural": "other",
    "Mid fallow": "non_crop",
    "Mixed Cereals": "cereals",
    "Banana": "fruits_nuts",
    "Sugarcane": "sugar",
    "Orange tree": "fruits_nuts",
    "Annual crop": "other",
    "Root": "root_tuber",
    "Vegetables": "vegetables_melons",
    "Cucurbit": "vegetables_melons",
}


def export_end_date(eos_date: date) -> datetime:
    eos_date = pd.to_datetime(eos_date)
    if (eos_date.month > EXPORT_END_MONTH) or (
        eos_date.month == EXPORT_END_MONTH and eos_date.day > EXPORT_END_DAY
    ):
        return datetime(int(eos_date.year + 1), EXPORT_END_MONTH, EXPORT_END_DAY)
    else:
        return datetime(int(eos_date.year), EXPORT_END_MONTH, EXPORT_END_DAY)


def load_jecam():
    df = geopandas.read_file(DATASET_PATH / "jecam")
    df = df.to_crs(LATLON_CRS)

    df["EOS"] = pd.to_datetime(df["EOS"])
    df = df[~df["EOS"].isnull()]
    df[RequiredColumns.EXPORT_END_DATE] = np.vectorize(export_end_date)(df.EOS)
    df = df[df[RequiredColumns.EXPORT_END_DATE].dt.year > 2017]

    df[RequiredColumns.LON] = df.geometry.centroid.x.values
    df[RequiredColumns.LAT] = df.geometry.centroid.y.values
    df = df.rename(columns={
        "AcquiDate": RequiredColumns.COLLECTION_DATE,
        "CropType1": NullableColumns.LABEL
    })

    df[NullableColumns.CLASSIFICATION_LABEL] = df.apply(
        lambda x: LABEL_TO_CLASSIFICATION[x[NullableColumns.LABEL]], axis=1
    )

    df[RequiredColumns.IS_CROP] = (df[NullableColumns.CLASSIFICATION_LABEL] != "non_crop").astype(
        int
    )

    df = df.reset_index(drop=True)
    df[RequiredColumns.INDEX] = df.index

    return df
