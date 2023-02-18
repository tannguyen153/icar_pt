from typing import Optional, List

class TrainerParams():
    ngpus: int =1
    epochs: int = 1
    steps: int = 1310720
    num_workers: int=10

class ModuleParams():
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    input_variables: int = 10  #state variables
    kernel_size: int = 64
    num_blocks: int=8 #num compute blocks
    n_outputs: int=2
    optim: str = 'adam'
    down_sampling_factor: int = 1
    state_var: str ='qr'
    #used to split the dataset into training and validation sets
    fold: int = 0
    n_splits: Optional[int] = 4
    seed: int = 0
    #print latest value of leanable parameters
    readout_freq: int = 1310720
    #activation: str = 'sigmoid'
    activation: str = 'swish'
    train_input: str = 'data/demo_input.nc'
    train_output: str = 'data/demo_output.nc'

if __name__ == '__main__':
    #just a test
    tparams= TrainerParams() 
    mparams= ModuleParams() 
    print('gpu', tparams.ngpus)
    print('epochs', tparams.epochs)
    print('lr', mparams.lr)
    print('weight_decay', mparams.weight_decay)
    print('batch_size', mparams.batch_size)
    print('optim', mparams.optim)
    print('fold', mparams.fold)
    print('n_splits', mparams.n_splits)
    print('seed', mparams.seed)
