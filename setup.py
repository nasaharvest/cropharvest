from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = "\n".join(
    [line for line in long_description.split("\n") if not line.startswith("<img")]
)

setup(
    name="cropharvest",
    description="Open source remote sensing dataset with benchmarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gabriel Tseng",
    author_email="gabrieltseng95@gmail.com",
    url="https://github.com/nasaharvest/cropharvest",
    version="0.6.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    packages=["cropharvest"] + [f"cropharvest.{f}" for f in find_packages("cropharvest")],
    install_requires=[
        "geopandas==0.9.0",
        "xarray>=0.16.2",
        "tqdm>=4.61.1",
        # h5py 3.7.0 breaks windows, see
        # https://github.com/h5py/h5py/issues/2110
        "h5py>=3.1.0,!=3.7.0",
        "rasterio>=1.2.6",
        "openpyxl>=2.5.9",
        "scikit-learn>=0.22.2",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
