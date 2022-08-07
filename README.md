# CropHarvest

CropHarvest is an open source remote sensing dataset for agriculture with benchmarks. It collects data from a variety of agricultural land use datasets and remote sensing products.

<img src="diagrams/labels_spatial_distribution.png" alt="Spatial distribution of labels" height="400px"/>

The dataset consists of **90,480** datapoints, of which **30,899** (34.2%) have multiclass labels. All other datapoints only have binary crop / non-crop labels.

**65,690** (73%) of these labels are paired with remote sensing and climatology data, specifically [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2), [Sentinel-1](https://sentinel.esa.int/web/sentinel/missions/sentinel-1/), the [SRTM Digital Elevation Model](https://cgiarcsi.community/data/srtm-90m-digital-elevation-database-v4-1/) and [ERA 5 climatology data](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5).

21 datasets are aggregated into CropHarvest - these are documented [here](https://github.com/nasaharvest/cropharvest/blob/main/datasets.md).

More details about CropHarvest and the benchmarks are available in [this paper](https://openreview.net/forum?id=JtjzUXPEaCu).

### Pipeline
The code in this repository

1) combines the constituent datasets into a single geoJSON file,
2) exports the associated satellite data from Earth Engine,
3) combines both datasets to create `(X,y)` training tuples and
4) exposes those tuples via a `Dataset` object.

The pipeline through which this happens is shown below:

<img src="diagrams/pipeline.svg" width="100%">

All blue boxes are associated with code in this repository. Anything green is data accessible via [Zenodo](https://zenodo.org/record/5828893). By default, the Zenodo data will get downloaded to the [data folder](https://github.com/nasaharvest/cropharvest/tree/main/data) - the data folder's [Readme](https://github.com/nasaharvest/cropharvest/blob/main/data/README.md) has more information about the exact structure of the data.

There are unique cases where you may need to use the `EarthEngineExporter` directly, these use cases are demonstrated in the [`demo_exporting_data.ipynb`](https://github.com/nasaharvest/cropharvest/blob/main/demo_exporting_data.ipynb) notebook.

### Installation
Linux and MacOS users can install the latest version of CropHarvest with the following command:
```bash
pip install cropharvest
```
Windows users must install the CropHarvest within a [conda](https://docs.conda.io/en/latest/miniconda.html) environment to ensure all dependencies are installed correctly:
```bash
conda install 'fiona>=1.5' 'rasterio>=1.2.6'
pip install cropharvest
```

### Getting started [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/cropharvest/blob/main/demo.ipynb)
See the [`demo.ipynb`](https://github.com/nasaharvest/cropharvest/blob/main/demo.ipynb) notebook for an example on how to download the data from [Zenodo](https://zenodo.org/record/5828893) and train a random forest against this data.

For more examples of models trained against this dataset, see the [benchmarks](https://github.com/nasaharvest/cropharvest/blob/main/benchmarks).

### Contributing
If you would like to contribute a dataset, please see the [contributing readme](https://github.com/nasaharvest/cropharvest/blob/main/contributing.md).

### License
CropHarvest has a [Creative Commons Attribution-ShareAlike 4.0 International](https://github.com/nasaharvest/cropharvest/blob/main/LICENSE.txt) license.

### Citation

If you use CropHarvest in your research, please use the following citation:
```
@inproceedings{
    tseng2021cropharvest,
    title={CropHarvest: A global dataset for crop-type classification},
    author={Gabriel Tseng and Ivan Zvonkov and Catherine Lilian Nakalembe and Hannah Kerner},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={https://openreview.net/forum?id=JtjzUXPEaCu}
}
```
