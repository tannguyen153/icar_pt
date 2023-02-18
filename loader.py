import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import num2date
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from collections import defaultdict
#from torchvision import transforms

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

