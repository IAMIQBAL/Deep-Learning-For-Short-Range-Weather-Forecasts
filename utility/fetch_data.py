import ee
import geemap
import os
from geemap import cartoee
import matplotlib.pyplot as plt

def fetchImages(lat, lng, regionCoords, startDate, endDate, visParams, band, saveDir, gifName='output.gif', w=4, h=2, dpi=50, ff='png'):
    point = ee.Geometry.Point(lng, lat)
    collection = (ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                 .filterBounds(point)
                 .filterDate(start, end)
                 .select(band))
    cartoee.get_image_collection_gif(
    ee_ic=collection,
    out_dir=os.path.expanduser(saveDir),
    out_gif=gifName,
    vis_params=visParams,
    region=regionCoords,
    fps=5,
    fig_size=(w, h),
    dpi_plot=dpi,
    file_format=ff,
    verbose=True,
)

# Temperature VisParams
visTEMP = {
  'min': 250,
  'max': 320,
  'palette': [
    '#000080', '#0000D9', '#4000FF', '#8000FF', '#0080FF', '#00FFFF', '#00FF80',
    '#80FF00', '#DAFF00', '#FFFF00', '#FFF500', '#FFDA00', '#FFB000', '#FFA400',
    '#FF4F00', '#FF2500', '#FF0A00', '#FF00FF'
  ]
}

# Precipitation VisParams
visPREC = {
    'min': 0,
    'max': 0.1,
    'palette': ['#FFFFFF', '#00FFFF', '#0080FF', '#DA00FF', '#FFA400', '#FF0000']
}


# 24 hrs
# 1982 to 2022 
start = '1989-09-27'
end = '1989-10-01'

band = 'total_precipitation' # temperature_2m OR total_precipitation
lon = 69.3451
lat = 30.3753
selection = [59.545898,23.155933,79.541016,37.439499]

fetchImages(lat, lon, selection, start, end, visPREC, band, 'precipitation/' + str(1989))