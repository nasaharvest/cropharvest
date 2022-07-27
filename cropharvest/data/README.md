# Data folder

This is the default data folder for CropHarvest, as defined in the [utils.py](../cropharvest/utils.py).
Unless otherwise specified, all files will be downloaded to here from Zenodo.

The following files can be downloaded:

#### 1. `labels.geojson`
```python
>>> import geopandas
>>> labels = geopandas.read_file("labels.geojson")
>>> labels.columns
Index(['harvest_date', 'planting_date', 'label', 'classification_label',
       'index', 'is_crop', 'lat', 'lon', 'dataset', 'collection_date',
       'export_end_date', 'is_test', 'geometry'],
      dtype='object')
```
There are two types of columns; `RequiredColumns` which must be filled for all rows, and `NullableColumns`, which can have null values (see [here](https://github.com/nasaharvest/cropharvest/blob/main/cropharvest/columns.py)).

##### 1.1. Required Columns
- `index` - the index of the row
- `is_crop` - a boolean indicating whether or not the point being described contains cropland or not (at the date described by `export_end_date`
- `lat` - the latitude of the point
- `lon` - the longitude of the point
- `dataset` - the [dataset](https://github.com/nasaharvest/cropharvest/blob/main/datasets.md) which the point comes from
- `collection_date` - the date at which the point was collected
- `export_end_date` - we collect a year of data for each point - this value defines the last month for which data is exported (and therefore the entire timeseries, since we will collect data for a year up to that point).
- `geometry` - the geometry of the point. This may be a polygon (in which case `lat`/`lon` will be the central point of that field) or a point
- `is_test` - a boolean indicating whether or not the point is part of the test data

###### 1.2. Nullable columns
- `harvest_date` - the harvest date of the crop described at the `lat`/`lon`
- `planting_date` - the planting date of the crop described at the `lat`/`lon`
- `label` - the label - this will be the higher level agricultural land cover label describing the land use at the `lat`/`lon` for the given `export_end_date`
- `classification_label` - the higher level classification of `label`, defined by the FAO's indicative crop classification (i.e. if a row has a `label="maize"`, then it would have `classification_label="cereals"`

#### 2. features
The features folder contains processed arrays (i.e. satellite data linked to a row from the `labels.geojson`) ready to be ingested by machine learning models. All features have the following naming convention: `{index}_{dataset}.h5` - where these two values are defined by the `labels.geojson`. So each feature is associated with a row in the `labels.geojson`.

We are currently [in the process](https://github.com/nasaharvest/cropharvest/pull/85) of changing this convention so that names are instead in a `f"min_lat={min_lat}_min_lon={min_lon}_max_lat={max_lat}_max_lon={max_lon}_dates={start_date}_{end_date}_all"` format.

#### 3. test_features
All test features are associated with an entry in the `TEST_REGIONS` or `TEST_DATASETS` in the [config.py](../cropharvest/config.py).

#### 4. eo_data
The eo_data folder contains [tif](https://en.wikipedia.org/wiki/TIFF) files exported from [Google Earth Engine](https://earthengine.google.com/). The naming format of these files is `{index}_{dataset}.tif`- where these two values are defined by the `labels.geojson`. So each tif file is associated with a row in the `labels.geojson`.

We are currently [in the process](https://github.com/nasaharvest/cropharvest/pull/85) of changing this convention so that names are instead in a `f"min_lat={min_lat}_min_lon={min_lon}_max_lat={max_lat}_max_lon={max_lon}_dates={start_date}_{end_date}_all"` format.
