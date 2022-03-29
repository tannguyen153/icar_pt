import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from loader import mapDataset, myIterableDataset, DataLoader
from torch.optim.optimizer import Optimizer
from sklearn.model_selection import KFold
from typing import Callable, List, Dict, Optional
from model import ICARModel
from params import ModuleParams, TrainerParams 
import netCDF4 as nc
from loss import Metrics, mae_loss

class ICARNet(pl.LightningModule):
    def __init__(self,tparams: TrainerParams, mparams: ModuleParams):
        super().__init__()        
        self.model = ICARModel(num_blocks=6)        
        self.best: float = float('inf')        

    def setup(self, stage: str):
        import pandas as pd
        import numpy as np
        import netCDF4 as nc

        fn= './data/example.nc'
        ds = nc.Dataset(fn)

        latitudes = ds.variables['lat'][:]
        longitudes = ds.variables['lon'][:]
        times = ds.variables['time'][:]
        times_grid, latitudes_grid, longtitudes_grid = [
            x.flatten() for x in np.meshgrid(times, latitudes, longitudes, indexing='ij')]
        df0 = pd.DataFrame({
            'time': times_grid,
            'latitude': latitudes_grid,
            'longitude': longtitudes_grid
        })

        
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
        df1 = pd.DataFrame({'precip': precip_grid})
        df = pd.concat([df0, df1], join = 'outer', axis=1)

        folds = KFold(
            n_splits= mparams.n_splits,
            random_state= mparams.seed,
            shuffle=True,
        )
        train_idx, val_idx = list(folds.split(df))[mparams.fold]
        self.train_dataset = mapDataset(df.iloc[train_idx])
        self.val_dataset = mapDataset(df.iloc[val_idx])

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
        batch=batch['latitude'].to(torch.float32)
        result = self.step(batch, prefix='train')
        return {
            'loss': result['train_loss'],
            **result,
        }

    def validation_step(self, batch, batch_idx):
        batch=batch['latitude'].to(torch.float32)
        result = self.step(batch, prefix='val')
        return {**result}

    def step(self, batch, prefix: str, model=None) -> Dict:
        if model is None:
            y_pred = self.forward(batch['time'])
        else:
            y_pred = model(batch['time'])
        y_true = batch['precip'].to(torch.float32)
        mae = mae_loss(y_pred, y_true)
        lmae = torch.log(mae)
        size = len(y_true)
        return {
            f'{prefix}_loss': lmae,
            f'{prefix}_mae': mae,
            f'{prefix}_size': size,
        }

#    def training_epoch_end(self, outputs):
#        return {}

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
        loss, mae = 0, 0
        total_size = 0
        for o in outputs:
            size = o[f'{prefix}_size']
            total_size += size
            loss += o[f'{prefix}_loss'] * size
            mae += o[f'{prefix}_mae'] * size
        loss = loss / total_size
        mae = mae / total_size
        return Metrics(
            lr= mparams.lr, 
            loss=loss,
            lmae=torch.log(mae),
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
    )
    net = ICARNet(tparams,mparams) 
    trainer.fit(net)


if __name__ == '__main__':
    tparams= TrainerParams()
    mparams= ModuleParams()
    train(tparams,mparams)
