# CropHarvest

CropHarvest is an open source remote sensing dataset for agriculture with benchmarks. It collects data from a variety of agricultural land use datasets and remote sensing products.

<img src="diagrams/labels_spatial_distribution.png" alt="Spatial distribution of labels" height="400px"/>

The dataset consists of **90,480** datapoints, of which **30,899** (34.2%) have multiclass labels. All other datapoints only have binary crop / non-crop labels.

**65,690** (73%) of these labels are paired with remote sensing and climatology data, specifically [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2), [Sentinel-1](https://sentinel.esa.int/web/sentinel/missions/sentinel-1/), the [SRTM Digital Elevation Model](https://cgiarcsi.community/data/srtm-90m-digital-elevation-database-v4-1/) and [ERA 5 climatology data](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5).

21 datasets are aggregated into CropHarvest - these are documented [here](https://github.com/nasaharvest/cropharvest/blob/main/datasets.md).

More details about CropHarvest and the benchmarks are available in [this paper](https://openreview.net/forum?id=JtjzUXPEaCu).

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

### Getting started
See the [`demo.ipynb`](https://github.com/nasaharvest/cropharvest/blob/main/demo.ipynb) notebook for an example on how to download the data from [Zenodo](https://zenodo.org/record/5533193) and train a random forest against this data.

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
