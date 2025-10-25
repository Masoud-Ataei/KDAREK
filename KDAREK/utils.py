import numpy as np
import torch

def Dataset(fx, n = 4, fix = True, a  = -2*np.pi, b  =  2*np.pi, seed = None):
    if not (seed is None):
        np.random.seed(seed)
    # Ti = np.linspace(-2*np.pi,2*np.pi,n)
    # a  = -2*np.pi
    # b  =  2*np.pi
    # Ti = a + b - (-a+b) * np.cos((2*j - 1) * np.pi / 2 / n) / 2
    if fix:
        Ti = np.linspace(a,b,n)
        Ttest = np.linspace(a,b,1000)
    else:
        Ti    = np.random.uniform(a,b,n)
        Ttest = np.random.uniform(a,b,1000)
    # Ti.sort()
    Yi    = fx(Ti)
    Ytest = fx(Ttest)

    # train the model
    device='cpu' # cpu or cuda
    dataset = {}
    dataset['train_input'] = torch.tensor(Ti    , dtype=torch.float32).reshape((-1,1)).to(device)
    dataset['test_input']  = torch.tensor(Ttest , dtype=torch.float32).reshape((-1,1)).to(device)

    dataset['train_label'] = torch.tensor(Yi    , dtype=torch.float32).reshape((-1,1)).to(device)
    dataset['test_label']  = torch.tensor(Ytest , dtype=torch.float32).reshape((-1,1)).to(device)
    return dataset
