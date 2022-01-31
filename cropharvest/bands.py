S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
ERA5_BANDS = ["temperature_2m", "total_precipitation"]
SRTM_BANDS = ["elevation", "slope"]

DYNAMIC_BANDS = S1_BANDS + S2_BANDS + ERA5_BANDS
STATIC_BANDS = SRTM_BANDS
