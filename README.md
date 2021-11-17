# CropHarvest

CropHarvest is an open source remote sensing dataset for agriculture with benchmarks. It collects data from a variety of agricultural land use datasets and remote sensing products.

<img src="diagrams/labels_spatial_distribution.png" alt="Spatial distribution of labels" height="400px"/>

### Installation

`cropharvest` can be pip installed by running `pip install cropharvest`

### Getting started
See the [`demo.ipynb`](demo.ipynb) notebook for an example on how to download the data from [Zenodo](https://zenodo.org/record/5533193) and train a random forest against this data.

For more examples of models trained against this dataset, see the [benchmarks](benchmarks).

More details about CropHarvest and the benchmarks are available in [this paper](https://openreview.net/forum?id=JtjzUXPEaCu).

### Contributing
If you would like to contribute a dataset, please see the [contributing readme](contributing.md).

### License
CropHarvest has a [Creative Commons Attribution-ShareAlike 4.0 International](LICENSE.txt) license.

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
