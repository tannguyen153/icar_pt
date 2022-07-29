import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from activations import swish
from loader import mapDataset, myIterableDataset, DataLoader
import sys

class Dense(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            activation=None,
    ):
        self.activation = activation
        self.iter = 0 
        super(Dense, self).__init__(in_features, out_features, bias)

    def forward(self, inputs):
        if self.iter%8192 ==0: 
            filename = "weights_iter"+str(self.iter)
            with open(filename, 'a') as f:
                with np.printoptions(threshold=np.inf, linewidth=np.inf):
                    sys.stdout = f 
                    print(np.transpose(self.weight.detach().numpy()))
                    print("\n")
                    print(np.transpose(self.bias.detach().numpy()))
                    print("\n \n")
        self.iter = self.iter+1
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
    def __init__(self, input_variables, inputSize, kernelSize, activation=None):
        super(encoding_block, self).__init__()
        #self.embedding = nn.Embedding(input_variables, inputSize, padding_idx=0)
        self.dense = Dense(input_variables, kernelSize, activation=activation)
        self.activation= activation
        self.input_variables = input_variables
        self.inputSize = inputSize
    def forward(self, inputs):
        qv, qr, qc, qi, ni, nr, qs, qg, temp, press = inputs
        #time = torch.nn.functional.normalize(time, dim=0)
        qv= torch.nn.functional.normalize(qv, dim=0)
        qr= torch.nn.functional.normalize(qr, dim=0)
        qc= torch.nn.functional.normalize(qc, dim=0)
        qi= torch.nn.functional.normalize(qi, dim=0)
        ni= torch.nn.functional.normalize(ni, dim=0)
        nr= torch.nn.functional.normalize(nr, dim=0)
        qs= torch.nn.functional.normalize(qs, dim=0)
        qg= torch.nn.functional.normalize(qg, dim=0)
        temp= torch.nn.functional.normalize(temp, dim=0)
        press= torch.nn.functional.normalize(press, dim=0)
        x = torch.cat((qv, qr, qc, qi, ni, nr, qs, qg, temp, press))
        x= x.detach().numpy()
        x= np.reshape(x, (self.input_variables, self.inputSize))
        x= np.transpose(x)
        x = self.dense(torch.tensor(x))
        return x

class comp_block(nn.Module):
    def __init__(self, inputSize, kernelSize, num_before_skip, num_after_skip, activation=None):
        super(comp_block, self).__init__()
        self.inputSize = inputSize
        self.kernelSize = kernelSize
        self.activation= activation

        self.layers_before_skip = nn.ModuleList([
            ResidualLayer(kernelSize, activation=activation, bias=True)
            for _ in range(num_before_skip)
        ])

        self.final_before_skip = Dense(kernelSize, kernelSize, activation=activation, bias=True)        

        self.layers_after_skip = nn.ModuleList([
            ResidualLayer(kernelSize, activation=activation, bias=True)
            for _ in range(num_after_skip)
        ])

    def forward(self, inputs):
        x = inputs
        #for layer in self.layers_before_skip:
        #    x = layer(x)
        x = self.final_before_skip(x)
        x= inputs + x
        #for layer in self.layers_after_skip:
        #    x = layer(x)
        return x

class output_block(nn.Module):
    def __init__(self, inputSize, kernelSize, activation=None):
        super(output_block, self).__init__()
        self.dense = Dense(kernelSize, 1, activation=activation)
        self.activation= activation
    def forward(self, inputs):
        return self.dense(inputs)


class ICARModel(nn.Module):
    def __init__(
        self,
        mparams,
        activation=swish
    ):
        super(ICARModel, self).__init__()
        self.num_blocks = mparams.num_blocks
        self.activation=activation
        self.encoding_block = encoding_block(
            input_variables=mparams.input_variables,
            inputSize=mparams.batch_size,            
            kernelSize=mparams.kernel_size,            
            activation=activation,
        )
        self.comp_blocks = nn.ModuleList([
            comp_block(
                inputSize=mparams.batch_size,
                kernelSize=mparams.kernel_size,
                num_before_skip=mparams.num_before_skip,
                num_after_skip=mparams.num_after_skip,
                activation=activation,
            )
            for _ in range(self.num_blocks)
        ])
        self.output_block = output_block(
            inputSize=mparams.batch_size,
            kernelSize=mparams.kernel_size,
            activation=activation,
        )
        

    def forward(self, inputs):
        x= inputs
        qv = inputs['qv']
        qr = inputs['qr']
        qc = inputs['qc']
        qi = inputs['qi']
        ni = inputs['ni']
        nr = inputs['nr']
        qs = inputs['qs']
        qg = inputs['qg']
        temp = inputs['temp']
        press = inputs['press']
        x= self.encoding_block([qv.to(torch.float32), qr.to(torch.float32), qc.to(torch.float32), qi.to(torch.float32), ni.to(torch.float32), nr.to(torch.float32), qs.to(torch.float32), qg.to(torch.float32), temp.to(torch.float32), press.to(torch.float32)])
        for i in range(self.num_blocks):
            x = self.comp_blocks[i](x)
        x= self.output_block(x)
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
