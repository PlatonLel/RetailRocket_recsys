import pandas as pd
import numpy as np
from collections import defaultdict
import random
from torch.utils.data import Dataset

class BPRDataset(Dataset):
    def __init__(self, train_data, num_items):
        self.train_data = train_data
        self.num_items = num_items
        self.user_positive_items = defaultdict(set)
        for u, i, _ in train_data:
            self.user_positive_items[u].add(i)
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        user, pos_item, weight = self.train_data[idx]
        neg_item = random.randint(0, self.num_items - 1)
        while neg_item in self.user_positive_items[user]:
            neg_item = random.randint(0, self.num_items - 1)
        return (
            np.int64(user),
            np.int64(pos_item),
            np.int64(neg_item),
            np.float32(weight)
        )