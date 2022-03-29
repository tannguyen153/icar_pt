
def swish(x, inplace: bool = False):
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())

def mish(x, inplace: bool = False):
    return x.mul(F.softplus(x).tanh())    

def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()

def tanh(x, inplace: bool = False):
    return x.tanh_() if inplace else x.tanh()
