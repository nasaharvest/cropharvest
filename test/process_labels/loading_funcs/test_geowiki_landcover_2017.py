from process_labels import datasets


def test_geowiki_landcover_2017():
    data = datasets.load("geowiki-landcover-2017")

    non_crops = data[data.mean_sumcrop <= 0.5]
    assert (non_crops.is_crop == 0).all()

    crops = data[data.mean_sumcrop > 0.5]
    assert (crops.is_crop == 1).all()
