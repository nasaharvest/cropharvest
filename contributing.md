# Contributing

If you would like to contribute a dataset, thank you! Please complete the following steps to do so:

### 1. Add the dataset to [process_labels](process_labels)
The first step is to open a pull request which adds the data to the `process_labels` folder in this repository. There are 3 places which will need to be changed:

* [ ] Add the raw data to the [raw data folder](process_labels/raw_data)
* [ ] Add a function which takes the raw data and returns a geojson of the appropriate format. Appropriate format means it has all of the required columns, and all the nullable columns possible. These columns are described in [columns.py](cropharvest/columns.py). The expected type of the columns is described (and tested) in [test_datasets.py](test/process_labels/test_datasets.py).
* [ ] Update the [DATASETS dict](process_labels/datasets.py) to include the newest dataset.

### 2. Recreate the data, and upload to Zenodo
You will need to recreate the `features`, and upload the new `labels.geojson` to Zenodo. Please contact Gabriel Tseng (gabrieltseng95@gmail.com) or Ivan Zvonkov (izvonkov@umd.edu) to do this.

### 3. Update the Zenodo link in CropHarvest
Add the new Zenodo identifier to the [cropharvest config](cropharvest/config.py). Because the python package is coupled to the data, this will also require a new release. Once again, please contact Gabriel or Ivan to do this.
