from cropharvest.countries import BBox


def test_add_boxes():
    a = BBox(min_lat=1, min_lon=1, max_lat=2, max_lon=2, name="a")
    b = BBox(min_lat=2, min_lon=2, max_lat=3, max_lon=3, name="b")

    ab = a + b
    assert ab.name == "a_b"
    assert ab.min_lon == ab.min_lat == 1
    assert ab.max_lon == ab.max_lat == 3
