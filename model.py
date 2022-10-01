import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from activations import swish
from activations import ReLU
from loader import mapDataset, myIterableDataset, DataLoader
import sys

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
        self.iter = 0 
        self.state_var= state_var
        self.readout_freq= readout_freq
        super(Dense, self).__init__(in_features, out_features, bias)

    def printWnB(self):
        filename= self.state_var+"_weights_iter"
        filename = filename+str(self.iter)
        with open(filename, 'a') as f:
            with np.printoptions(threshold=np.inf, linewidth=np.inf):
                sys.stdout = f 
                print(np.transpose(self.weight.detach().numpy()))
                print("\n")
                print(np.transpose(self.bias.detach().numpy()))
                print("\n \n")

    def forward(self, inputs):
        y = super(Dense, self).forward(inputs)
        if(self.activation == 'relu'): 
            y = ReLU(y)
        elif(self.activation == 'swish'): 
            y = swish(y, inplace=True)
        if self.iter % self.readout_freq ==0: 
            self.printWnB()
        self.iter = self.iter+1
        return y

class ResidualLayer(nn.Module):
    def __init__(self, kernel_size, **kwargs):
        super(ResidualLayer, self).__init__()
        self.dense_1 = Dense(kernel_size, kernel_size, **kwargs)
        self.dense_2 = Dense(kernel_size, kernel_size, **kwargs)
    def forward(self, inputs):
        x = inputs + self.dense_2(self.dense_1(inputs))
        return x

class encoding_block(nn.Module):    
    def __init__(self, input_variables, input_batch_size, kernel_size, activation="", state_var="", readout_freq=1):
        super(encoding_block, self).__init__()
        self.dense = Dense(input_variables, kernel_size, activation=activation, state_var=state_var, readout_freq=readout_freq)
        self.activation= activation
        self.input_variables = input_variables
        self.input_batch_size = input_batch_size

    def forward(self, inputs):
        qv, qr, qc, qi, ni, nr, qs, qg, temp, press = inputs
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
        x= np.reshape(x, (self.input_variables, self.input_batch_size))
        x= np.transpose(x)
        x = self.dense(torch.tensor(x))
        return x

class comp_block(nn.Module):
    def __init__(self, input_batch_size, kernel_size, num_before_skip, num_after_skip, activation="", state_var="", readout_freq=1):
        super(comp_block, self).__init__()
        self.input_batch_size = input_batch_size
        self.kernel_size = kernel_size
        self.activation= activation

        self.layers_before_skip = nn.ModuleList([
            ResidualLayer(kernel_size, activation=activation, bias=True, state_var=state_var, readout_freq=readout_freq)
            for _ in range(num_before_skip)
        ])

        self.final_before_skip = Dense(kernel_size, kernel_size, activation=activation, bias=True, state_var=state_var, readout_freq=readout_freq)        

        self.layers_after_skip = nn.ModuleList([
            ResidualLayer(kernel_size, activation=activation, bias=True, state_var=state_var, readout_freq=readout_freq)
            for _ in range(num_after_skip)
        ])

    def forward(self, inputs):
        x = inputs
        #for layer in self.layers_before_skip:
        #    x = layer(x)
        x = self.final_before_skip(x)
        x = inputs + x
        #for layer in self.layers_after_skip:
        #    x = layer(x)
        return x

class output_block(nn.Module):
    def __init__(self, input_batch_size, kernel_size, activation="", state_var="", readout_freq=1):
        super(output_block, self).__init__()
        self.dense = Dense(kernel_size, 1, activation=activation, state_var=state_var, readout_freq=readout_freq)
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
                num_before_skip=mparams.num_before_skip,
                num_after_skip=mparams.num_after_skip,
                activation=self.activation,
                state_var= mparams.state_var,
                readout_freq= mparams.readout_freq
            )
            for _ in range(self.num_blocks)
        ])
        self.output_block = output_block(
            input_batch_size=mparams.batch_size,
            kernel_size=mparams.kernel_size,
            activation=self.activation,
            state_var= mparams.state_var,
            readout_freq= mparams.readout_freq
        )

    def forward(self, inputs):
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
        x     = self.encoding_block([qv.to(torch.float32), qr.to(torch.float32), qc.to(torch.float32), qi.to(torch.float32), ni.to(torch.float32), nr.to(torch.float32), qs.to(torch.float32), qg.to(torch.float32), temp.to(torch.float32), press.to(torch.float32)])
        for i in range(self.num_blocks):
            x = self.comp_blocks[i](x)
        x= self.output_block(x)
        return x

if __name__ == '__main__':
    print("We will write some tests later")
