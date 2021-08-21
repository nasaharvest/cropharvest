from setuptools import setup, find_packages

long_description = """
CropHarvest is an open source remote sensing dataset for agriculture with benchmarks. 
It collects data a variety of agricultural land use datasets and remote sensing products.
"""

setup(
    name="cropharvest",
    description="Open source remote sensing dataset with benchmarks",
    long_description=long_description,
    author="Gabriel Tseng",
    author_email="gabrieltseng95@gmail.com",
    url="https://github.com/nasaharvest/cropharvest",
    version="0.0.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["cropharvest"] + [f"cropharvest.{f}" for f in find_packages("cropharvest")],
    install_requires=[
        "geopandas==0.9.0",
        "xarray>=0.16.2",
        "earthengine-api>=0.1.271",
        "tqdm>=4.61.1",
        "h5py>=3.1.0",
        "rasterio>=1.2.6",
        "openpyxl>=2.5.9",
        "scikit-learn>=0.22.2",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
