import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from activations import sigmoid,swish
from activations import ReLU
from loader import mapDataset, myIterableDataset, DataLoader
import sys
from typing import Dict


class Dense(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            activation="",
            state_var="",
            readout_freq=1
    ):
        self.activation = activation
        self.state_var= state_var
        self.readout_freq= readout_freq
        super(Dense, self).__init__(in_features, out_features, bias)
        self.layer= nn.Linear(in_features, out_features, bias, dtype=float)

    def forward(self, inputs):
        y=self.layer(inputs)
        if(self.activation == 'relu'): 
            y = ReLU(y)
        elif(self.activation == 'swish'): 
            y = swish(y, inplace=True)
        elif(self.activation == 'sigmoid'):
            y = sigmoid(y, inplace=True)
        return y

class encoding_block(nn.Module):    
    def __init__(self, input_variables, input_batch_size, kernel_size, activation="", state_var="", readout_freq=1):
        super(encoding_block, self).__init__()
        self.dense = Dense(input_variables, kernel_size, activation=activation, state_var=state_var, readout_freq=readout_freq)
        self.activation= activation
        self.input_variables = input_variables
        self.input_batch_size = input_batch_size

    def forward(self, inputs: Dict[str, torch.Tensor]):
        qv    = inputs['qv']
        qr    = inputs['qr']
        qc    = inputs['qc']
        qi    = inputs['qi']
        ni    = inputs['ni']
        nr    = inputs['nr']
        qs    = inputs['qs']
        qg    = inputs['qg']
        temp  = inputs['temp']
        press = inputs['press']
        x = torch.cat((qv, qr, qc, qi, ni, nr, qs, qg, temp, press))
        x= torch.reshape(x, (self.input_variables, self.input_batch_size))
        x= torch.transpose(x, 0, 1)
        x= self.dense(x)
        return x

class comp_block(nn.Module):
    def __init__(self, input_batch_size, kernel_size, activation="", state_var="", readout_freq=1):
        super(comp_block, self).__init__()
        self.input_batch_size = input_batch_size
        self.kernel_size = kernel_size
        self.activation= activation
        self.dense = Dense(kernel_size, kernel_size, activation=activation, bias=True, state_var=state_var, readout_freq=readout_freq)        

    def forward(self, inputs):
        x = inputs
        x = self.dense(x)
        x = inputs + x
        return x

class output_block(nn.Module):
    def __init__(self, input_batch_size, kernel_size, num_output=1, activation="", state_var="", readout_freq=1):
        super(output_block, self).__init__()
        self.dense = Dense(kernel_size, num_output, activation=activation, state_var=state_var, readout_freq=readout_freq)
        self.activation= activation
    def forward(self, inputs):
        return self.dense(inputs)


class ICARModel(nn.Module):
    def __init__(
        self,
        mparams
    ):
        super(ICARModel, self).__init__()
        self.num_blocks = mparams.num_blocks
        self.activation = mparams.activation
            
        self.encoding_block = encoding_block(
            input_variables=mparams.input_variables,
            input_batch_size=mparams.batch_size,            
            kernel_size=mparams.kernel_size,            
            activation=self.activation,
            state_var= mparams.state_var,
            readout_freq= mparams.readout_freq
        )
        self.comp_blocks = nn.ModuleList([
            comp_block(
                input_batch_size=mparams.batch_size,
                kernel_size=mparams.kernel_size,
                activation=self.activation,
                state_var= mparams.state_var,
                readout_freq= mparams.readout_freq
            )
            for _ in range(self.num_blocks)
        ])
        self.output_block = output_block(
            input_batch_size=mparams.batch_size,
            kernel_size=mparams.kernel_size,
            num_output=mparams.n_outputs,
            activation=self.activation,
            state_var= mparams.state_var,
            readout_freq= mparams.readout_freq
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        x     = self.encoding_block(inputs)
        for index, layer in enumerate (self.comp_blocks):
            x= layer(x)
        x= self.output_block(x)
        return x

if __name__ == '__main__':
    print("We will write some tests later")
