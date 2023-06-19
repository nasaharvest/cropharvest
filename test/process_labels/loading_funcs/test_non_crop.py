from process_labels import datasets


def test_kenya_non_crop():
    data = datasets.load("kenya-non-crop")
    assert (data.is_crop == 0).all()


def test_mali_non_crop():
    data = datasets.load("mali-non-crop")
    assert (data.is_crop == 0).all()


def test_brazil_non_crop():
    data = datasets.load("brazil-non-crop")
    assert (data.is_crop == 0).all()
