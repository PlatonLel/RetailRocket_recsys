import torch
from torch.utils.data import Dataset

class BPRDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        user, pos_item, neg_item, pos_cat, neg_cat, pos_prop_type, neg_prop_type, pos_prop_value, neg_prop_value = self.triplets[idx]
        return {
            'user_id': torch.tensor(user, dtype=torch.long),
            'pos_item_id': torch.tensor(pos_item, dtype=torch.long),
            'neg_item_id': torch.tensor(neg_item, dtype=torch.long),
            'pos_cat': torch.tensor(pos_cat, dtype=torch.long),
            'neg_cat': torch.tensor(neg_cat, dtype=torch.long),
            'pos_prop_type': torch.tensor(pos_prop_type, dtype=torch.long),
            'neg_prop_type': torch.tensor(neg_prop_type, dtype=torch.long),
            'pos_prop_value': torch.tensor(pos_prop_value, dtype=torch.long),
            'neg_prop_value': torch.tensor(neg_prop_value, dtype=torch.long),
        }