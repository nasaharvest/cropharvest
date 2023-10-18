import pandas as pd
import geopandas
from datetime import datetime

from cropharvest.config import EXPORT_END_MONTH, EXPORT_END_DAY
from cropharvest.columns import RequiredColumns

from .utils import LATLON_CRS
from ..utils import DATASET_PATH


def raw_from_ceo():
    # this is the code used to go from the CEO (unanonymised) labels
    # to the csv in `raw_data`. While we keep this here for posterity,
    # the original data is not uploaded to git

    def combine_csvs(format_name: str, year: int, name_2=None):
        assert str(year) in format_name
        if name_2 is not None:
            df = join_on_agreement(pd.read_csv(format_name), pd.read_csv(name_2))
        else:
            df = join_on_agreement(
                pd.read_csv(format_name.format(set_val=1)),
                pd.read_csv(format_name.format(set_val=2)),
            )
        df["year"] = year
        return df

    def join_on_agreement(a, b):
        def clean_df(a):
            crop_column = None
            for crop_column_name in [
                "Does this pixel contain active cropland?",
                "Does this point fall within active cropland?",
                "Does this point lie on active cropland?",
                "Does this point contain active cropland?",
            ]:
                if crop_column_name in a.columns:
                    crop_column = crop_column_name
            assert crop_column is not None
            a = a[["lon", "lat", crop_column, "collection_time"]]
            a = a.rename(columns={crop_column: "crop", "collection_time": "collection_date"})
            return a

        a = clean_df(a)
        b = clean_df(b)
        joined = a.merge(b, on=["lat", "lon"], how="inner", suffixes=("", "_y"))
        return joined[joined.crop == joined.crop_y][["lon", "lat", "crop", "collection_date"]]

    dfs = []
    for filename, name_2, year in [
        (
            "ceo-Liaoning-2019-April---November-(Set-{set_val})-sample-data-2022-06-27.csv",
            None,
            2019,
        ),
        (
            "ceo-Liaoning-2019-April---November-(Set-{set_val})-sample-data-2022-06-27.csv",
            None,
            2019,
        ),
        (
            "ceo-Jilin-2017-(Set-2)-sample-data-2021-04-15.csv",
            "ceo-Jilin-2017-(Set-1)-sample-data-2021-04-19.csv",
            2017,
        ),
        ("ceo-HLJ-2019-(Set-{set_val})---v3-sample-data-2022-01-21.csv", None, 2019),
        ("ceo-Heilongjiang-2016-(Set-{set_val})---v2-sample-data-2022-01-21.csv", None, 2016),
        ("ceo-Heilongjiang-2017-(Set-{set_val})---v2-sample-data-2022-01-21.csv", None, 2017),
        ("ceo-Heilongjiang-2018-(Set-{set_val})-sample-data-2021-10-26.csv", None, 2018),
        (
            "ceo-Jilin-2016-(April-November)-(Set-{set_val})-v2-sample-data-2022-06-27.csv",
            None,
            2016,
        ),
        (
            "ceo-Jilin-2018-(April---November)-(Set-{set_val})-sample-data-2022-06-27.csv",
            None,
            2018,
        ),
        ("ceo-Jilin-2019-April---November-(Set-{set_val})-sample-data-2022-06-27.csv", None, 2019),
        (
            "ceo-Liaoning-2016-April---November-(Set-{set_val})-sample-data-2022-06-27.csv",
            None,
            2016,
        ),
        ("ceo-Liaoning-2017-(Set-{set_val})-sample-data-2021-04-09.csv", None, 2017),
        (
            "ceo-Liaoning-2018-April---November-(Set-{set_val})-sample-data-2022-06-27.csv",
            None,
            2018,
        ),
        (
            "ceo-Liaoning-2019-April---November-(Set-{set_val})-sample-data-2022-06-27.csv",
            None,
            2019,
        ),
    ]:
        dfs.append(combine_csvs(filename, year, name_2))
    return pd.concat(dfs)


def load_china() -> geopandas.GeoDataFrame:
    df = pd.read_csv(DATASET_PATH / "china" / "combined_and_anonymised_points.csv")
    gdf = geopandas.GeoDataFrame(
        data=df, geometry=geopandas.points_from_xy(df.lon, df.lat), crs=LATLON_CRS
    )
    gdf = df.reset_index(drop=True)
    gdf[RequiredColumns.IS_CROP] = df.apply(lambda x: 1 if x.crop == "Crop" else 0)
    gdf[RequiredColumns.INDEX] = df.index
    gdf[RequiredColumns.EXPORT_END_DATE] = df.apply(
        lambda x: datetime(x.year, EXPORT_END_MONTH, EXPORT_END_DAY), axis=1
    )
    gdf[RequiredColumns.COLLECTION_DATE] = pd.to_datetime(df.collection_date)
    return gdf
