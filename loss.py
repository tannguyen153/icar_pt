import torch

class Metrics:
    lr: float   #learning rate
    loss: float #loss 
    def __init__(self, lr, loss):
        self.lr=lr
        self.loss=loss

def mae_loss(y_pred, y_true):
    err = torch.abs(y_true - y_pred)
    mae = err.mean()
    return mae

def mse_loss(y_pred, y_true):
    err = torch.square(y_true - y_pred).mean()
    return err 
