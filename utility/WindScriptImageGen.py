import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import os
from tqdm.auto import tqdm

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_minmax_date(data):
    dmin = data['time'].values.min()
    dmax = data['time'].values.max()

    sec0 = np.datetime64(0, 's')
    sec1 = np.timedelta64(1, 's')
    
    xmin = datetime(1970, 1, 1) + timedelta(seconds=(dmin - sec0) / sec1)
    xmax = datetime(1970, 1, 1) + timedelta(seconds=(dmax - sec0) / sec1) + timedelta(days=1)

    return xmin.date(), xmax.date()
        
def write_data(data, dates, cmap, v_min, v_max, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in dates:
        print(i[0].year)
        for single_date in daterange(i[0], i[1]):
            for j in range(24):
                point = single_date.strftime("%Y%m%dT") + '{:02}'.format(j) # {:02} for max 2 digits
                y = out_dir + '/' + point + '.png'
                fig, ax = plt.subplots(figsize=(1, 0.8))
                data.sel(time=point).plot(ax=ax, cmap=cmap, add_colorbar=False, vmin=v_min, vmax=v_max)
                plt.title('')
                plt.axis('off')
                fig.savefig(y, bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close(fig)

UMIN = -11.4
UMAX = 11.4
CMAP = 'gray'
OUTDIR = 'year'

# Done: 9
for i in range(12, 15):
    # i = 8
    path = 'Wind ' + str(i) + '.nc'
    # print(path)
    DS = xr.open_dataset(path)['v10']
    dmin, dmax = get_minmax_date(DS)
    # dmin = date(1956, 1, 7)
    # print(dmin, dmax)
    dates = np.array([[date(dmin.year, dmin.month, dmin.day), date(dmax.year, dmax.month, dmax.day)]]) # Exclusive dates
    write_data(DS, dates, 'gray', UMIN, UMAX, 'DataSetWindU/1970-2022-V/')
    DS.close()