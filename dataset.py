import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, item_sequences=None,
                 category_sequences=None,
                 property_type_sequences=None,
                 property_value_sequences=None,
                 event_types_sequences_int=None,
                 event_types_sequences_str=None,
                 item_weights=None,
                 seq_length=5):
        self.samples = []

        event_types_sequences_for_model = event_types_sequences_int if event_types_sequences_int else [[] for _ in range(len(item_sequences))]
        event_types_sequences_for_eval = event_types_sequences_str if event_types_sequences_str else [[] for _ in range(len(item_sequences))]
        
        for idx in range(len(item_sequences)):
            seq_items = item_sequences[idx]
            seq_cats = category_sequences[idx]
            seq_prop_types = property_type_sequences[idx]
            seq_prop_values = property_value_sequences[idx]
            seq_weights = item_weights[idx] if item_weights else [1.0] * len(seq_items)
            seq_event_types_int = event_types_sequences_for_model[idx]
            seq_event_types_str = event_types_sequences_for_eval[idx]
            
            for j in range(len(seq_items) - 1):
                item_input = seq_items[:j+1][-seq_length:]
                category_input = seq_cats[:j+1][-seq_length:]
                prop_type_input = seq_prop_types[:j+1][-seq_length:]
                prop_value_input = seq_prop_values[:j+1][-seq_length:]
                event_type_input = seq_event_types_int[:j+1][-seq_length:]
                padding_size = seq_length - len(item_input)
                if padding_size > 0:
                    item_input = [0] * padding_size + item_input
                    category_input = [0] * padding_size + category_input
                    prop_type_input = [0] * padding_size + prop_type_input
                    prop_value_input = [0] * padding_size + prop_value_input
                    event_type_input = [0] * padding_size + event_type_input
                item_target = seq_items[j+1]
                category_target = seq_cats[j+1]
                target_event_type_str = seq_event_types_str[j+1]
                
                sample = {
                    'item_seq': item_input,
                    'category_seq': category_input,
                    'prop_type_seq': prop_type_input,
                    'event_type_seq': event_type_input,
                    'prop_value_seq': prop_value_input, 
                    'item_target': item_target,
                    'category_target': category_target,
                    'target_event_type_str': target_event_type_str,
                    # 'target_weight': seq_weights[j+1]
                }
                
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'item_seq': torch.tensor(sample['item_seq'], dtype=torch.long),
            'category_seq': torch.tensor(sample['category_seq'], dtype=torch.long),
            'event_type_seq': torch.tensor(sample['event_type_seq'], dtype=torch.long),
            'item_target': torch.tensor(sample['item_target'], dtype=torch.long),
            'category_target': torch.tensor(sample['category_target'], dtype=torch.long),
            # 'event_type_target': torch.tensor(sample['target_event_type'], dtype=torch.long),
            'prop_type_seq': torch.tensor(sample['prop_type_seq'], dtype=torch.long),
            'prop_value_seq': torch.tensor(sample['prop_value_seq'], dtype=torch.long),
            # 'target_weight': torch.tensor(sample['target_weight'], dtype=torch.float),
            'target_event_type_str': sample['target_event_type_str']
        }