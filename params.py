from typing import Optional, List

class TrainerParams():
    ngpus: int =2
    epochs: int = 8
    num_workers: int=10

class ModuleParams():
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    input_variables: int = 10  #state variables
    num_blocks: int=4 #num compute blocks
    num_before_skip: int=1
    num_after_skip: int=1
    optim: str = 'adam'
    down_sampling_factor: int = 32
    #used to split the dataset into training and validation sets
    fold: int = 0
    n_splits: Optional[int] = 4
    seed: int = 0

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
