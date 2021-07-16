from setuptools import setup


setup(
    name="cropharvest",
    description="Open source remote sensing dataset with benchmarks",
    packages=["cropharvest"],
    install_requires=[
        "geopandas",
        "xarray",
        "earthengine-api" "tqdm",
        "h5py",
        "rasterio",
        "openpyxl",
        "sklearn",
    ],
)
