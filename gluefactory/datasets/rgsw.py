import torch
import os
from torch.utils.data import Dataset, DataLoader
from gluefactory.datasets.base_dataset import collate

class RGSW(Dataset):
    def __init__(self, num_samples=1, path='/usr/local/home/rsoussan/glue-factory/data/rgsw'):
        self.num_samples = num_samples 
        self.path = path

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.path, f'rgsw_data_{idx}.pt'))

    def get_dataloader(self, batch_size=1, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
