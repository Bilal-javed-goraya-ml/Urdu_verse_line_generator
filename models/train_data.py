import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import Dataset
import pandas as pd

# Dataset
class MasraDataset(Dataset):
    def __init__(self, csv_path, vocab):
        self.data = pd.read_csv(csv_path)
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq = self.vocab.encode(self.data.iloc[idx]['input'])
        output_seq = self.vocab.encode(self.data.iloc[idx]['output'])
        return torch.tensor(input_seq), torch.tensor(output_seq)