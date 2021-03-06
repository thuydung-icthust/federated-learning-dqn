from torch._C import device
from utils.loader import CustomDataset
from torch.utils.data import DataLoader
from torch import nn
import torch
import copy


class Client(object):
    def __init__(
        self,
        idx,
        dataset,
        list_idx_sample,
        list_abiprocess,
        batch_size,
        lr,
        epochs,
        mu,
        valid_dataset,
        algorithm="fedprox",
    ):
        super().__init__()
        self.train_dataloader = DataLoader(CustomDataset(dataset, list_idx_sample[idx]), batch_size=batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(CustomDataset(dataset, valid_dataset.indices), batch_size=batch_size, shuffle=False)
        # self.valid_dataloader = valid_dataset
        
        self.algorithm = algorithm
        self.lr = lr
        self.eps = epochs
        self.n_samples = len(list_idx_sample[idx])
        self.mu = mu
        self.abiprocess = list_abiprocess[idx]
