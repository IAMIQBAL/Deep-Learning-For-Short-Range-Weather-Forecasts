import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from datetime import date, timedelta

def daysDelta(start_date, end_date):
    return (end_date - start_date).days

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def createDataset(dates, inDir, daysCount):
#     delta = (end_date - start_date).days
    dataset = np.zeros(shape=(daysCount, 24, 74, 104), dtype=np.int16) # Change the first dimension before executing which is days
    animCount = 0
    for i in dates:
        for single_date in daterange(i[0], i[1]):
            anim = np.zeros(shape=(24, 74, 104), dtype=np.int16)
            for j in range(24):
                imgName = single_date.strftime("%Y%m%dT") + '{:02}.png'.format(j)
                image = cv2.imread(inDir + imgName)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                anim[j] = image
            dataset[animCount] = anim
            animCount += 1
    return dataset

# Change start_date and end_date accordingly
start_date1 = date(2019, 3, 1)
end_date1 = date(2019, 5, 1)
start_date2 = date(2019, 7, 1)
end_date2 = date(2019, 10, 1)
start_date3 = date(2019, 12, 1)
end_date3 = date(2020, 1, 1)
start_date4 = date(2018, 12, 1)
end_date4 = date(2019, 3, 1)
start_date5 = date(2018, 7, 1)
end_date5 = date(2018, 10, 1)

dates = np.array([[date(1981, 7, 1), date(1981, 10, 1)], 
                 [date(1982, 7, 1), date(1982, 10, 1)], 
                 [date(1983, 7, 1), date(1983, 10, 1)], 
                 [date(1984, 7, 1), date(1984, 10, 1)], 
                 [date(1985, 7, 1), date(1985, 10, 1)], 
                 [date(1986, 7, 1), date(1986, 10, 1)], 
                 [date(1987, 7, 1), date(1987, 10, 1)], 
                 [date(1988, 7, 1), date(1988, 10, 1)], 
                 [date(1989, 7, 1), date(1989, 10, 1)], 
                 [date(1990, 7, 1), date(1990, 10, 1)], 
                 [date(1991, 7, 1), date(1991, 10, 1)], 
                 [date(1992, 7, 1), date(1992, 10, 1)], 
                 [date(1993, 7, 1), date(1993, 10, 1)]])

s = 0
for i in dates:
    s += daysDelta(i[0], i[1])

data = createDataset(dates, 
                     'PrecipitationCropped/',
                     s)

np.save('prec81-93', data)