# Release Steps
Releasing a new package involves uploading new data to Zenodo and releasing the new version of the package on Pypi.
## Zenodo Upload Instructions
1. Create tars if needed
    ```bash
    cd cropharvest/data
    tar -czf eo_data.tar.gz eo_data
    tar -czf features.tar.gz features
    cd ../..
    ```
2. Set zenodo token 
    ```bash
    export ZENODO_TOKEN=<your zenodo token>
    ```

3. Go to the newest version of CropHarvest on Zenodo and click "New Version" and leave the page open (note the sizes of the files).
4. Copy the deposit id from URL (after /deposit/) ie. https://zenodo.org/deposit/5567762 and set it as an environment variable:
    ```bash
    export DEPOSITION_ID=5567762
    ```

5. Upload all or one of the files
    ```bash
    ./zenodo_upload.sh $DEPOSITION_ID /home/ubuntu/cropharvest/data/eo_data.tar.gz
    ./zenodo_upload.sh $DEPOSITION_ID /home/ubuntu/cropharvest/data/features.tar.gz
    ./zenodo_upload.sh $DEPOSITION_ID /home/ubuntu/cropharvest/data/labels.geojson
    ```
6. Go back to the Zenodo page, refresh and verify that the new tar.gz size is updated.
7. Click publish


/Users/izvonkov/nasaharvest/cropharvest/dist/cropharvest-0.2.0.tar.gz
## Python Package Release
1. Update the `DATASET_VERSION_ID` in [cropharvest/config](cropharvest/config.py)
2. Update the package version in setup.py
3. Set the package version as an environment variable:
    ```bash
    export PACKAGE_VERSION=0.2.0
    ```
3. Build the package
    ```bash
    python -m build
    ```
4. Install utility for publishing packages
    ```bash
    pip install twine
    ```
5. Check the distribution
    ```bash
    twine check \
    dist/cropharvest-${PACKAGE_VERSION}.tar.gz \
    dist/cropharvest-${PACKAGE_VERSION}-py3-none-any.whl
    ```
6. Uploads to test pypi
    ```bash
    twine upload --repository testpypi \
    dist/cropharvest-${PACKAGE_VERSION}.tar.gz \
    dist/cropharvest-${PACKAGE_VERSION}-py3-none-any.whl
    ```
7. Test package from testpypi 
    ```bash
    pip install -i https://test.pypi.org/simple/ cropharvest
    python
    >>> from cropharvest.datasets import CropHarvest
    >>> CropHarvest.create_benchmark_datasets("data")
    ```
7.  Uploads to real pypi
    ```bash
    twine upload \
    dist/cropharvest-${PACKAGE_VERSION}.tar.gz \
    dist/cropharvest-${PACKAGE_VERSION}-py3-none-any.whl
    ```

## Tag and Github Release
1. After merging the PR with the above changes to the package, tag the release with the new version number
    ```bash
    git tag v${PACKAGE_VERSION}
    git push --tags
    ```

2. Create a release on Github.