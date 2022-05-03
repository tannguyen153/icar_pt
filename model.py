import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from activations import swish
from loader import mapDataset, myIterableDataset, DataLoader

class Dense(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            activation=None,
    ):
        self.activation = activation
        super(Dense, self).__init__(in_features, out_features, bias)

    def forward(self, inputs):
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = self.activation(y)
        return y

class ResidualLayer(nn.Module):
    def __init__(self, featureSize, **kwargs):
        super(ResidualLayer, self).__init__()
        self.dense_1 = Dense(featureSize, featureSize, **kwargs)
        self.dense_2 = Dense(featureSize, featureSize, **kwargs)
    def forward(self, inputs):
        x = inputs + self.dense_2(self.dense_1(inputs))
        return x

class encoding_block(nn.Module):    
    def __init__(self, input_variables=5, input_size=1, activation=None):
        super(encoding_block, self).__init__()
        self.embedding = nn.Embedding(input_variables, input_size, padding_idx=0)
        self.dense = Dense(input_size * input_variables, input_size, activation=activation)
    def forward(self, inputs):
        time, level, lat, lon, qr = inputs
        time = torch.nn.functional.normalize(time, dim=0)
        lat= torch.nn.functional.normalize(lat, dim=0)
        lon= torch.nn.functional.normalize(lon, dim=0)
        x = torch.cat((time, level, lat, lon, qr))
        x = self.dense(x)
        return x

class comp_block(nn.Module):
    def __init__(self, inputSize, num_before_skip, num_after_skip, activation=None):
        super(comp_block, self).__init__()
        self.inputSize = inputSize

        self.layers_before_skip = nn.ModuleList([
            ResidualLayer(inputSize, activation=activation, bias=True)
            for _ in range(num_before_skip)
        ])

        self.final_before_skip = Dense(inputSize, inputSize, activation=activation, bias=True)        

        self.layers_after_skip = nn.ModuleList([
            ResidualLayer(inputSize, activation=activation, bias=True)
            for _ in range(num_after_skip)
        ])

    def forward(self, inputs):
        x1 = inputs
        for layer in self.layers_before_skip:
            x1 = layer(x1)
        x1 = self.final_before_skip(x1)
        x= inputs + x1
        for layer in self.layers_after_skip:
            x = layer(x)
        return x

class ICARModel(nn.Module):
    def __init__(
        self,
        input_variables=5,
        inputSize=1,
        num_blocks=1,
        num_before_skip=1,
        num_after_skip=1,
        activation=swish
    ):
        super(ICARModel, self).__init__()
        self.num_blocks = num_blocks
        self.encoding_block = encoding_block(
            input_variables,
            inputSize,            
            activation=activation,
        )
        self.comp_blocks = nn.ModuleList([
            comp_block(
                inputSize=1,
                num_before_skip=num_before_skip,
                num_after_skip=num_after_skip,
                activation=activation,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, inputs):
        x= inputs
        time= inputs['time']
        level= inputs['level']
        lat=  inputs['latitude']
        lon=  inputs['longitude']
        inputData= inputs['input']
        x= self.encoding_block([time.to(torch.float32), level.to(torch.float32), lat.to(torch.float32), lon.to(torch.float32), inputData.to(torch.float32)])
        for i in range(self.num_blocks):
            x = self.comp_blocks[i](x)
        return x

if __name__ == '__main__':
    #test the model
    import pandas as pd
    import numpy as np
    import netCDF4 as nc

    fn= './data/example.nc'
    ds = nc.Dataset(fn)
    latitudes = ds.variables['lat'][:]
    longitudes = ds.variables['lon'][:]
    levels = ds.variables['plev'][:]
    times = ds.variables['time'][:]
    times_grid, latitudes_grid, longtitudes_grid, level_grid = [
        x.flatten() for x in np.meshgrid(times, latitudes, longitudes, levels, indexing='ij')]
    df = pd.DataFrame({
        'time': times_grid,
        'latitude': latitudes_grid,
        'longitude': longtitudes_grid,
        'level': level_grid
    })
    dataset = mapDataset(df)
    inputs= DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    m = ICARModel(
        num_blocks=6,
    )

    for batch in inputs:
        input1=batch['input'].to(torch.float32)
        output = m(input1)# torch.from_numpy(batch['level'].numpy().astype(np.float32)))
        break #just a test, early exit
