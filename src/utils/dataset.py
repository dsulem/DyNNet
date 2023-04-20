import os
import pandas as pd
from torch.utils.data import Dataset
import pickle

class GraphPairsDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(data_dir + '/labels.p', 'rb') as f:
            self.pairs_labels = pickle.load(f)

    def __len__(self):
        return len(self.pairs_labels)

    def __getitem__(self, idx):
        with open(self.data_dir + '/data.p', 'rb') as f:
            triplet = pickle.load(f)[idx]
        label = self.pairs_labels[idx]

        return (triplet[0], triplet[1]), label