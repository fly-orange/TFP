''' Convert parms to data or model'''
import torch
import torch.optim as optim
from model.gman import GMAN
from model.gman2 import GMAN2
from model.gwn import gwnet
from model.gt import GraphTrans
from utils.utils import masked_mae 

data_files = {
    'METR-LA':'data/METR-LA/metr-la.h5',
    'PEMS-BAY': 'data/PEMS-BAY/pems-bay.h5'
}

nodes_objects = {
    'METR-LA': 207,
    'PEMS-BAY': 325
}

model_objects = {
    'GMAN': GMAN,
    'GWN': gwnet,
    'GT': GraphTrans,
    'GMAN2': GMAN2
}

criterion_objects = {
    'MSE': torch.nn.MSELoss(),
    'L1': torch.nn.L1Loss(),
    'MAE': masked_mae
}

opt_objects = {
    'Adam': optim.Adam,
    'SGD': optim.SGD
}

sche_object ={
    'step': optim.lr_scheduler.StepLR,
    'exp': optim.lr_scheduler.ExponentialLR,
    'cos': optim.lr_scheduler.CosineAnnealingLR
}