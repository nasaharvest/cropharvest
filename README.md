# CropHarvest

CropHarvest is an open source remote sensing dataset for agriculture with benchmarks. It collects data a variety of agricultural land use datasets and remote sensing products.

<img src="diagrams/labels_spatial_distribution.png" alt="Spatial distribution of labels" height="400px"/>

### Installation

`cropharvest` can be pip installed by running `pip install -e .` where `.` is the path to the directory containing [`setup.py`](setup.py).

For development, all CI tests are run using a virtual environment. The same can be set up locally using the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### Getting started
See the [`demo.ipynb`](demo.ipynb) notebook for an example on how to download the data from [Zenodo](https://zenodo.org/record/5153470) and train a random forest against this data.

For more examples of models trained against this dataset, see the [benchmarks](benchmarks).

### Contributing
If you would like to contribute a dataset, please see the [contributing readme](contributing.md).

### License
CropHarvest has a [Creative Commons Attribution-ShareAlike 4.0 International](LICENSE.txt) license.
