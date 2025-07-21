import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import date, timedelta
import os

# Change start_date and end_date accordingly
year = 0
start_date = date(year, 7, 1) # Included Date
end_date = date(year, 10, 1) # Excluded Date

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def crop(start_date, end_date, inDir, outDir):
#     print(inDir)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    for single_date in daterange(start_date, end_date):
        for j in range(24):
            imgName = single_date.strftime("%Y%m%dT") + '{:02}.png'.format(j)
#             print(imgName)
            image = cv2.imread(inDir + imgName)
            cropped = image[6:80, 6:110]
            cv2.imwrite((outDir + imgName), cropped)