from typing import List


class Columns:
    @classmethod
    def tolist(cls) -> List[str]:
        # we don't want to return the magic functions
        return [
            value
            for name, value in vars(cls).items()
            if not (name.startswith("__") or name == "date_columns")
        ]

    @classmethod
    def date_columns(cls) -> List[str]:
        raise NotImplementedError


class RequiredColumns(Columns):

    INDEX = "dataset_index"
    IS_CROP = "is_crop"
    LAT = "lat"
    LON = "lon"
    DATASET = "dataset"
    COLLECTION_DATE = "collection_date"
    EXPORT_END_DATE = "export_end_date"
    GEOMETRY = "geometry"
    IS_TEST = "is_test"

    @classmethod
    def date_columns(cls) -> List[str]:
        return [cls.COLLECTION_DATE, cls.EXPORT_END_DATE]


class NullableColumns(Columns):
    HARVEST_DATE = "harvest_date"
    PLANTING_DATE = "planting_date"
    LABEL = "label"
    CLASSIFICATION_LABEL = "classification_label"

    @classmethod
    def date_columns(cls) -> List[str]:
        return [cls.HARVEST_DATE, cls.PLANTING_DATE]


class EngColumns:
    """
    Some columns uniquely created & used by the labels
    as loaded by the Engineer
    """

    FEATURES_FILENAME = "features_filename"
    FEATURES_PATH = "features_path"
    EXISTS = "feature_exists"
    TIF_FILEPATHS = "tif_path"
