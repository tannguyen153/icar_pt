import torch
import torch.nn as nn
import pytorch_lightning as pl
from activations import swish
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
from params import ModuleParams, TrainerParams 
import netCDF4 as nc
import pandas as pd
import numpy as np
from loss import Metrics, mse_loss
import sys


class ICARNet(pl.LightningModule):
    def __init__(self,tparams: TrainerParams, mparams: ModuleParams):
        super().__init__()        
        self.model = ICARModel(mparams, activation=swish)        
        self.best: float = float('inf')        

    def setup(self, stage: str):
        inputData= './data/training_input.nc'
        outputData= './data/training_output.nc'

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

        ds_o = nc.Dataset(outputData)
        qr_o = ds_o.variables['qr'][:,:,:,:]
        qr_sum= qr_o+qr #sum of positive values is zero only if both operands are zero
        num_rows=np.count_nonzero(qr_sum)
        nz_idx= np.nonzero(qr_sum) 
        qv= qv[nz_idx]
        qr= qr[nz_idx]
        qc= qc[nz_idx]
        qi= qi[nz_idx]
        ni= ni[nz_idx]
        nr= nr[nz_idx]
        qs= qs[nz_idx]
        qg= qg[nz_idx]
        temp= temp[nz_idx]
        press= press[nz_idx]
        qr_o= qr_o[nz_idx]

        columnList=['qv', 'qr', 'qc', 'qi', 'ni', 'nr', 'qs', 'qg', 'temp', 'press', 'output']
        npArray = np.zeros(shape=(num_rows,len(columnList)))
        mergedArray= np.column_stack((qv, qr, qc, qi, ni, nr, qs, qg, temp, press, qr_o))
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

    def validation_step(self, batch, batch_idx):
        result = self.step(batch, prefix='val')
        self.log('val_loss', result['val_loss'],on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)#, sync_dist_op='mean')
        return {**result}

    def step(self, batch, prefix: str, model=None) -> Dict:
        if model is None:
            y_pred = self.forward(batch)
        else:
            y_pred = model(batch)
        y_true = (batch['output']-batch['qr']).to(torch.float32)
        mse= mse_loss(y_pred, y_true)
        size = len(y_true)
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


def train(tparams: TrainerParams, mparams: ModuleParams):
    seed_everything(mparams.seed)
    trainer = pl.Trainer(
        max_epochs=tparams.epochs, 
        #accelerator="gpu", devices=[0] #comment this line if train on the CPUs
        #accelerator="ddp", gpus=tparams.ngpus #comment this line if train on the CPUs
    )
    net = ICARNet(tparams, mparams) 
    #if mparams.printout:
    #    with open('networkArch', 'w') as f:
    #        with np.printoptions(threshold=np.inf):
    #            sys.stdout = f
    #            print(net)
    trainer.fit(net)


if __name__ == '__main__':
    tparams= TrainerParams()
    mparams= ModuleParams()
    train(tparams,mparams)
