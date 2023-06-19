import ee
from datetime import date
import math

from cropharvest.bands import S2_BANDS as BANDS
from .utils import date_to_string


# These are algorithm settings for the cloud filtering algorithm
image_collection = "COPERNICUS/S2"

# Ranges from 0-1.Lower value will mask more pixels out.
# Generally 0.1-0.3 works well with 0.2 being used most commonly
cloudThresh = 0.2
# Height of clouds to use to project cloud shadows
cloudHeights = [200, 10000, 250]
# Sum of IR bands to include as shadows within TDOM and the
# shadow shift method (lower number masks out less)
irSumThresh = 0.3
ndviThresh = -0.1
# Pixels to reduce cloud mask and dark shadows by to reduce inclusion
# of single-pixel comission errors
erodePixels = 1.5
dilationPixels = 3

# images with less than this many cloud pixels will be used with normal
# mosaicing (most recent on top)
cloudFreeKeepThresh = 3


def get_single_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    dates = ee.DateRange(
        date_to_string(start_date),
        date_to_string(end_date),
    )

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()
    imgC = ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region)

    imgC = (
        imgC.map(lambda x: x.clip(region))
        .map(lambda x: x.set("ROI", region))
        .map(computeS2CloudScore)
        .map(projectShadows)
        .map(computeQualityScore)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    # has to be double to be compatible with the sentinel 1 imagery, which is in
    # float64
    cloudFree = mergeCollection(imgC).toDouble()

    return cloudFree


def rescale(img, exp, thresholds):
    return (
        img.expression(exp, {"img": img})
        .subtract(thresholds[0])
        .divide(thresholds[1] - thresholds[0])
    )


def computeQualityScore(img):
    score = img.select(["cloudScore"]).max(img.select(["shadowScore"]))

    score = score.reproject("EPSG:4326", None, 20).reduceNeighborhood(
        reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(5), optimization="boxcar"
    )

    score = score.multiply(-1)

    return img.addBands(score.rename("cloudShadowScore"))


def computeS2CloudScore(img):
    toa = img.select(BANDS).divide(10000)

    toa = toa.addBands(img.select(["QA60"]))

    # ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A',
    #  'B9',          'B10', 'B11','B12']
    # ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2',
    #  'waterVapor', 'cirrus','swir1', 'swir2']);

    # Compute several indicators of cloudyness and take the minimum of them.
    score = ee.Image(1)

    # Clouds are reasonably bright in the blue and cirrus bands.
    score = score.min(rescale(toa, "img.B2", [0.1, 0.5]))
    score = score.min(rescale(toa, "img.B1", [0.1, 0.3]))
    score = score.min(rescale(toa, "img.B1 + img.B10", [0.15, 0.2]))

    # Clouds are reasonably bright in all visible bands.
    score = score.min(rescale(toa, "img.B4 + img.B3 + img.B2", [0.2, 0.8]))

    # Clouds are moist
    ndmi = img.normalizedDifference(["B8", "B11"])
    score = score.min(rescale(ndmi, "img", [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = img.normalizedDifference(["B3", "B11"])
    score = score.min(rescale(ndsi, "img", [0.8, 0.6]))

    # Clip the lower end of the score
    score = score.max(ee.Image(0.001))

    # score = score.multiply(dilated)
    score = score.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(5))

    return img.addBands(score.rename("cloudScore"))


def projectShadows(image):
    meanAzimuth = image.get("MEAN_SOLAR_AZIMUTH_ANGLE")
    meanZenith = image.get("MEAN_SOLAR_ZENITH_ANGLE")

    cloudMask = image.select(["cloudScore"]).gt(cloudThresh)

    # Find dark pixels
    darkPixelsImg = image.select(["B8", "B11", "B12"]).divide(10000).reduce(ee.Reducer.sum())

    ndvi = image.normalizedDifference(["B8", "B4"])
    waterMask = ndvi.lt(ndviThresh)

    darkPixels = darkPixelsImg.lt(irSumThresh)

    # Get the mask of pixels which might be shadows excluding water
    darkPixelMask = darkPixels.And(waterMask.Not())
    darkPixelMask = darkPixelMask.And(cloudMask.Not())

    # Find where cloud shadows should be based on solar geometry
    # Convert to radians
    azR = ee.Number(meanAzimuth).add(180).multiply(math.pi).divide(180.0)
    zenR = ee.Number(meanZenith).multiply(math.pi).divide(180.0)

    # Find the shadows
    def getShadows(cloudHeight):
        cloudHeight = ee.Number(cloudHeight)

        shadowCastedDistance = zenR.tan().multiply(cloudHeight)  # Distance shadow is cast
        x = azR.sin().multiply(shadowCastedDistance).multiply(-1)  # /X distance of shadow
        y = azR.cos().multiply(shadowCastedDistance).multiply(-1)  # Y distance of shadow
        return image.select(["cloudScore"]).displace(
            ee.Image.constant(x).addBands(ee.Image.constant(y))
        )

    shadows = ee.List(cloudHeights).map(getShadows)
    shadowMasks = ee.ImageCollection.fromImages(shadows)
    shadowMask = shadowMasks.mean()

    # Create shadow mask
    shadowMask = dilatedErossion(shadowMask.multiply(darkPixelMask))

    shadowScore = shadowMask.reduceNeighborhood(
        **{"reducer": ee.Reducer.max(), "kernel": ee.Kernel.square(1)}
    )

    return image.addBands(shadowScore.rename(["shadowScore"]))


def dilatedErossion(score):
    # Perform opening on the cloud scores

    def erode(img, distance):
        d = (
            img.Not()
            .unmask(1)
            .fastDistanceTransform(30)
            .sqrt()
            .multiply(ee.Image.pixelArea().sqrt())
        )
        return img.updateMask(d.gt(distance))

    def dilate(img, distance):
        d = img.fastDistanceTransform(30).sqrt().multiply(ee.Image.pixelArea().sqrt())
        return d.lt(distance)

    score = score.reproject("EPSG:4326", None, 20)
    score = erode(score, erodePixels)
    score = dilate(score, dilationPixels)

    return score.reproject("EPSG:4326", None, 20)


def mergeCollection(imgC):
    return imgC.qualityMosaic("cloudShadowScore")
