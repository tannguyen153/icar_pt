import torch
import torch.nn as nn
import pytorch_lightning as pl
from activations import sigmoid,swish
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from loader import mapDataset, myIterableDataset, DataLoader
from torch.optim.optimizer import Optimizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from typing import Callable, List, Dict, Optional
from model import ICARModel
from params_qr_demo import ModuleParams, TrainerParams 
import netCDF4 as nc
import pandas as pd
import numpy as np
from loss import Metrics, mse_loss
import sys
import nexport
import os


class ICARNet(pl.LightningModule):
    def __init__(self,tparams: TrainerParams, mparams: ModuleParams):
        super().__init__()        
        self.model = ICARModel(mparams)        
        self.best: float = float('inf')        

    def setup(self, stage: str):
        inputData= mparams.train_input
        print('training input: ', inputData)
        outputData= mparams.train_output
        print('training output: ', outputData)

        ds = nc.Dataset(inputData)
        qv = ds.variables['qv'][:,:,:,:]
        qr = ds.variables['qr'][:,:,:,:]
        qc = ds.variables['qc'][:,:,:,:]
        qi = ds.variables['qi'][:,:,:,:]
        ni = ds.variables['ni'][:,:,:,:]
        nr = ds.variables['nr'][:,:,:,:]
        qs = ds.variables['qs'][:,:,:,:]
        qg = ds.variables['qg'][:,:,:,:]
        temp = ds.variables['temperature'][:,:,:,:]
        press = ds.variables['pressure'][:,:,:,:]
        time = ds.variables['time'][:]
        dt= time[:-1]
        dt1=time[1:]
        dt=dt1-dt
        dt= dt*24*60*60 #convert to seconds

        ds_o = nc.Dataset(outputData)
        qr_o = ds_o.variables['qr'][:,:,:,:]

        #keep original input and output
        self.qr= qr
        self.qr_o=qr_o

        #scaling inputs
        scaleFactor=1.0/120

        qr_sum= qr_o+qr #sum of positive values is zero only if both operands are zero
        qr_sum= qr_sum[:-1,:,:,:]        
        num_rows=np.count_nonzero(qr_sum)
        nz_idx= np.nonzero(qr_sum) 

        qv= qv[nz_idx]*scaleFactor
        qr= qr[nz_idx]*scaleFactor
        qc= qc[nz_idx]*scaleFactor
        qi= qi[nz_idx]*scaleFactor
        ni= ni[nz_idx]*scaleFactor
        nr= nr[nz_idx]*scaleFactor
        qs= qs[nz_idx]*scaleFactor
        qg= qg[nz_idx]*scaleFactor
        temp= temp[nz_idx]*scaleFactor
        press= press[nz_idx]*scaleFactor
        qr_o= qr_o[nz_idx]*scaleFactor
        dti=nz_idx[0]
        dt=dt[dti]        

        columnList=['qv', 'qr', 'qc', 'qi', 'ni', 'nr', 'qs', 'qg', 'temp', 'press', 'output', 'dt', 't', 'lat', 'lon']
        npArray = np.zeros(shape=(num_rows,len(columnList)))
        mergedArray= np.column_stack((qv, qr, qc, qi, ni, nr, qs, qg, temp, press, qr_o, dt, nz_idx[0], nz_idx[2], nz_idx[3]))
        #down sample the dataset by the given factor
        mask = np.random.choice([False, True], len(mergedArray), p=[1-1/mparams.down_sampling_factor, 1/mparams.down_sampling_factor])
        mergedArray=mergedArray[mask]
        #discard some samples so that the number of samples is divisible to the batch size
        datasetSize= len(mergedArray)- len(mergedArray)%mparams.batch_size
        mergedArray=mergedArray[0:datasetSize]
        #create a dataframe based on the dataset
        df = pd.DataFrame(mergedArray, columns=columnList)
        #split the dataset to training and validation sets
        folds = KFold(
            n_splits= mparams.n_splits,
            random_state= mparams.seed,
            shuffle=True,
        )
        train_idx, val_idx = list(folds.split(df))[mparams.fold]
        train_idx= train_idx[0:len(train_idx)-len(train_idx)%(mparams.batch_size*tparams.ngpus)]
        val_idx  = val_idx  [0:len(val_idx)-len(val_idx)%(mparams.batch_size*tparams.ngpus)]
        self.train_dataset = mapDataset(df.iloc[train_idx])
        self.val_dataset = mapDataset(df.iloc[val_idx])
        self.inference_dataset = mapDataset(df.iloc[val_idx])
        
        print("Traning size", len(train_idx))
        print("Validating size", len(val_idx))

    def on_train_start(self) -> None:        
        super(ICARNet, self).on_train_start()

    def optimizer_step(        
            self,            
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            optimizer_closure=0,
            on_tpu = False,            
            using_native_amp = False,         
            using_lbfgs = False,
    ) -> None: 
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx,optimizer_closure)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        result = self.step(batch, prefix='train')
        self.log('train_loss', result['train_loss'],on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)#, sync_dist_op='mean')
        return {
            'loss': result['train_loss'],
            **result,
        }

    def predict_step(self, batch, batch_idx):
        y_pred      = self.model(batch)
        dt =         batch['dt']        
        y_ref  = y_pred.clone()
        org_qr_o = self.qr_o 
        org_qr = self.qr
        dt= batch['dt']
        for i in range (len(y_ref)):
            t=int(batch['t'][i].item())
            lat= int(batch['lat'][i].item())
            lon= int(batch['lon'][i].item())
            y_ref[i]= torch.tensor(org_qr_o[t, :, lat, lon]- org_qr[t, :, lat, lon])/dt[i]

        with open("qr_predict_demo", 'a') as f:
            with np.printoptions(threshold=np.inf, linewidth=np.inf):
                sys.stdout = f
                torch.set_printoptions(precision=16,sci_mode=True)
                print ("qv input", batch['qv'])
                print ("qr input", batch['qr'])
                print ("qc input", batch['qc'])
                print ("qi input", batch['qi'])
                print ("ni input", batch['ni'])
                print ("nr input", batch['nr'])
                print ("qs input", batch['qs'])
                print ("qg input", batch['qg'])
                print ("temperature input", batch['temp'])
                print ("press input", batch['press'])
                print ("qr output", batch['output'])
                print ("dt", batch['dt'])
                print ("qr predicted", y_pred)
                print ("qr true output", y_ref)
                sys.stdout = sys.__stdout__

    def validation_step(self, batch, batch_idx):
        result = self.step(batch, prefix='val')
        self.log('val_loss', result['val_loss'],on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)#, sync_dist_op='mean')
        return {**result}

    def step(self, batch, prefix: str) -> Dict:
        y_pred = self.forward(batch)
        y_ref  = y_pred.clone()
        org_qr_o = self.qr_o
        org_qr = self.qr        
        dt= batch['dt']
        for i in range (len(y_ref)):
            t=int(batch['t'][i].item())
            lat=int(batch['lat'][i].item())
            lon=int(batch['lon'][i].item())
            y_ref[i]= torch.tensor(org_qr_o[t, :, lat, lon]- org_qr[t, :, lat, lon])/dt[i]

        mse= mse_loss(y_pred, y_ref)
        size = len(y_ref)
        return {
            f'{prefix}_loss': torch.sqrt(mse),
            f'{prefix}_size': size,
        }

    #def training_epoch_end(self, outputs):
    #    return {}

    def validation_epoch_end(self, outputs):
        metrics = self.__collect_metrics(outputs, 'val')
        if metrics.loss < self.best:
            self.best = metrics.loss
        return {
            'progress_bar': {
                'val_loss': metrics.loss,
                'best': self.best,
            },
            'val_loss': metrics.loss,
        }            

    def __collect_metrics(self, outputs: List[Dict], prefix: str) -> Metrics:
        loss, mse = 0, 0
        total_size = 0
        for o in outputs:
            size = o[f'{prefix}_size']
            total_size += size
            loss += o[f'{prefix}_loss'] * size
        loss = loss / total_size
        return Metrics(
            lr= mparams.lr, 
            loss=loss,
        )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size= mparams.batch_size,
            shuffle=True,
            num_workers= tparams.num_workers,
            pin_memory=True,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size= mparams.batch_size,
            shuffle=False,
            num_workers= tparams.num_workers,
            pin_memory=True,
        )
    def predict_dataloader(self):
        return DataLoader(
            self.inference_dataset,
            batch_size= mparams.batch_size,
            shuffle=False,
            num_workers= tparams.num_workers,
            pin_memory=True,
        )
    def configure_optimizers(self):
        if mparams.optim == 'adam':
            optim = torch.optim.Adam 
        else:
            raise Exception('Optim Not Supported}')
        opt = optim(
            self.model.parameters(),
            lr= mparams.lr,
            weight_decay= mparams.weight_decay,
        )
        return [opt]

def printWnB(layer: nn.Linear, filename: str):
    with open(filename, 'a') as f:
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            sys.stdout = f
            print(np.transpose(layer.weight.detach().numpy()))
            print("\n")
            print(np.transpose(layer.bias.detach().numpy()))
            print("\n \n")
            sys.stdout = sys.__stdout__

def readout(net: ICARNet, filename: str):
    model   = net.model
    ecBlock = model.encoding_block
    cBlocks = model.comp_blocks
    oBlock  = model.output_block
    printWnB (ecBlock.dense, filename)
    for b in cBlocks:
        printWnB (b.dense, filename)
    printWnB (oBlock.dense, filename)
    

def train(tparams: TrainerParams, mparams: ModuleParams):
    seed_everything(mparams.seed)
    trainer = pl.Trainer(
        max_epochs=tparams.epochs, 
        max_steps=tparams.steps,
        #accelerator="gpu", devices=[0] #comment this line if train on the CPUs
        #accelerator="ddp", gpus=tparams.ngpus #comment this line if train on the CPUs
    )
    net = ICARNet(tparams, mparams) 
    trainer.fit(net)

    #read out weights and biases of the trained network
    #filename= mparams.state_var+"_WnB"
    #readout(net, filename)
    #filename=filename+"_json"
    #nexport.export(model=net, filetype="json_exp", filename=filename)

    #now use the trained model to do inference on the whole dataset
    trainer.predict(model=net)

if __name__ == '__main__':
    tparams= TrainerParams()
    mparams= ModuleParams()
    train(tparams,mparams)
