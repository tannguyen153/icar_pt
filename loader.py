import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import num2date
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

class mapDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row.to_dict()
    def __len__(self):
        return len(self.df)

class myIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        return iter(self.data)

if __name__ == '__main__':
    #just to test the netcdf file loader
    fn= './data/example.nc'
    ds = nc.Dataset(fn)
    latitudes = ds.variables['lat'][:]
    longitudes = ds.variables['lon'][:]
    times = ds.variables['time'][:]
    precips = ds.variables['precip']
    print(np.ma.count(precips))

    precip_grid= np.zeros(len(times)* len(latitudes) * len(longitudes), dtype=np.double)
    for k in range(len(times)):
        koff= k*len(longitudes)*len(latitudes)
        print("Plane offset", koff)
        for j in range(len(latitudes)):
            joff= koff+j*len(longitudes)
            for i in range(len(longitudes)):
                kji= joff+i
                if np.ma.is_masked(ds.variables['precip'][k,j,i])==False:
                    precip_grid[kji]= ds.variables['precip'][k,j,i]
                else: precip_grid[kji]= 0
    df0 = pd.DataFrame({'precip': precip_grid})

    times_grid, latitudes_grid, longtitudes_grid = [ 
        x.flatten() for x in np.meshgrid(times, latitudes, longitudes, indexing='ij')]
    df1 = pd.DataFrame({
        'time': times_grid,
        'latitude': latitudes_grid,
        'longitude': longtitudes_grid,
        })

    df = pd.concat([df0, df1], join = 'outer', axis=1)        
    df.to_csv("inputs_time_lat_lon.csv", index=False)
