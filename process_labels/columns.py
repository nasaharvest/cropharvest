from typing import List


class Columns:
    @classmethod
    def tolist(cls) -> List[str]:
        # we don't want to return the magic functions
        return [value for name, value in vars(cls).items() if not name.startswith("__")]


class RequiredColumns(Columns):

    INDEX = "index"
    IS_CROP = "is_crop"
    LAT = "lat"
    LON = "lon"
    DATASET = "dataset"
    COLLECTION_DATE = "collection_date"
    EXPORT_END_DATE = "export_end_date"
    GEOMETRY = "geometry"
    IS_TEST = "is_test"


class NullableColumns(Columns):
    HARVEST_DATE = "harvest_date"
    PLANTING_DATE = "planting_date"
    LABEL = "label"
    CLASSIFICATION_LABEL = "classification_label"
