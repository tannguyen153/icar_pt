import torch

class Metrics:
    lr: float   #learning rate
    loss: float #loss 
    lmae: float #log of the mean absolute error   
    def __init__(self, lr, loss, lmae):
        self.lr=lr
        self.loss=loss
        self.lmae=lmae

def mae_loss(y_pred, y_true):
    err = torch.abs(y_true - y_pred)
    mae = err.mean()
    return mae
