import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, 
                 item_sequences=None,
                 category_sequences=None,
                 property_type_sequences=None,
                 property_value_sequences=None,
                 seq_length=5):
        self.samples = []

        for idx in range(len(item_sequences)):
            seq_items = item_sequences[idx]
            seq_cats = category_sequences[idx]
            seq_prop_types = property_type_sequences[idx]
            seq_prop_values = property_value_sequences[idx]
            
            for j in range(len(seq_items) - 1):
                item_input = seq_items[:j+1][-seq_length:]
                category_input = seq_cats[:j+1][-seq_length:]
                prop_type_input = seq_prop_types[:j+1][-seq_length:]
                prop_value_input = seq_prop_values[:j+1][-seq_length:]
                padding_size = seq_length - len(item_input)
                if padding_size > 0:
                    item_input = [0] * padding_size + item_input
                    category_input = [0] * padding_size + category_input
                    prop_type_input = [0] * padding_size + prop_type_input
                    prop_value_input = [0] * padding_size + prop_value_input
                item_target = seq_items[j+1]
                category_target = seq_cats[j+1]
                
                sample = {
                    'item_seq': item_input,
                    'category_seq': category_input,
                    'prop_type_seq': prop_type_input,
                    'prop_value_seq': prop_value_input, 
                    'item_target': item_target,
                    'category_target': category_target,
                }
                
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'item_seq': torch.tensor(sample['item_seq'], dtype=torch.long),
            'category_seq': torch.tensor(sample['category_seq'], dtype=torch.long),
            'item_target': torch.tensor(sample['item_target'], dtype=torch.long),
            'category_target': torch.tensor(sample['category_target'], dtype=torch.long),
            'prop_type_seq': torch.tensor(sample['prop_type_seq'], dtype=torch.long),
            'prop_value_seq': torch.tensor(sample['prop_value_seq'], dtype=torch.long),
        }