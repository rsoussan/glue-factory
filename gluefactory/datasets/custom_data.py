import torch
import glob 
import os
from torch.utils.data import Dataset, DataLoader
from gluefactory.datasets.base_dataset import collate

class CustomData(Dataset):
    def __init__(self, path):
        file_path = os.path.join(path, '*.pt')
        self.num_samples = len(glob.glob(file_path)) 
        # Assumes {x}_0.pt exists in path
        self.prefix = os.path.basename(glob.glob(file_path)[0]).rsplit("_0.pt", 1)[0] 
        self.path = path

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.path, f'{self.prefix}_{idx}.pt'))
    
    @classmethod
    def get_dataloader(cls, path, batch_size=1, shuffle=False):
        dataset = cls(path) 
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
