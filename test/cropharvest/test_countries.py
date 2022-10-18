from cropharvest.countries import BBox, get_country_bbox


def test_add_boxes():
    a = BBox(min_lat=1, min_lon=1, max_lat=2, max_lon=2, name="a")
    b = BBox(min_lat=2, min_lon=2, max_lat=3, max_lon=3, name="b")

    ab = a + b
    assert ab.name == "a_b"
    assert ab.min_lon == ab.min_lat == 1
    assert ab.max_lon == ab.max_lat == 3


def test_add_boxes_missing_name():
    a = BBox(min_lat=1, min_lon=1, max_lat=2, max_lon=2, name="a")
    b = BBox(min_lat=2, min_lon=2, max_lat=3, max_lon=3, name=None)

    ab = a + b
    assert ab.name == "a"
    assert ab.min_lon == ab.min_lat == 1
    assert ab.max_lon == ab.max_lat == 3


def test_largest_country():
    france = get_country_bbox("France", largest_only=True)
    assert france[0].name == "France_0"  # the biggest polygon for France
