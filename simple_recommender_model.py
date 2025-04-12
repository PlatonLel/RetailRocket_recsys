import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
from collections import defaultdict
import bisect
class SequenceDataset(Dataset):
    def __init__(self, item_sequences, category_sequences, item_weights=None, item_event_types=None, seq_length=5):
        self.samples = []
        
        for idx in range(len(item_sequences)):
            seq_items = item_sequences[idx]
            seq_cats = category_sequences[idx]
            seq_weights = item_weights[idx] if item_weights else [1.0] * len(seq_items)
            seq_event_types = item_event_types[idx] if item_event_types else ["view"] * len(seq_items)
            
            for j in range(len(seq_items) - 1):
                item_input = seq_items[:j+1][-seq_length:]
                category_input = seq_cats[:j+1][-seq_length:]
                
                padding_size = seq_length - len(item_input)
                if padding_size > 0:
                    item_input = [0] * padding_size + item_input
                    category_input = [0] * padding_size + category_input
                
                item_target = seq_items[j+1]
                category_target = seq_cats[j+1]
                target_weight = seq_weights[j+1]
                target_event_type = seq_event_types[j+1]
                
                sample = {
                    'item_seq': item_input,
                    'category_seq': category_input,
                    'item_target': item_target,
                    'category_target': category_target,
                    'target_weight': target_weight,
                    'target_event_type': target_event_type,
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
            'target_weight': torch.tensor(sample['target_weight'], dtype=torch.float),
            'target_event_type': sample['target_event_type']
        }

class SequenceModel(nn.Module):
    def __init__(self, num_items, num_categories,  embedding_dim=64):
        super(SequenceModel, self).__init__()


        
        self.item_embeddings = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.category_embeddings = nn.Embedding(num_categories, embedding_dim, padding_idx=0)
        
        self.item_gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.category_gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        
        self.category_predictor = nn.Linear(embedding_dim, num_categories)
        self.item_predictor = nn.Linear(embedding_dim * 2, num_items)
        
    def forward(self, item_sequences, category_sequences):
        batch_size = item_sequences.size(0)
        
        item_emb = self.item_embeddings(item_sequences)
        category_emb = self.category_embeddings(category_sequences)
        
        _, item_hidden = self.item_gru(item_emb)
        item_hidden = item_hidden.view(batch_size, -1)
        
        _, category_hidden = self.category_gru(category_emb)
        category_hidden = category_hidden.view(batch_size, -1)
        
        category_logits = self.category_predictor(category_hidden)
        combined_hidden = torch.cat([item_hidden, category_hidden], dim=1)
        item_logits = self.item_predictor(combined_hidden)
        
        return category_logits, item_logits
    
    def calculate_loss(self, category_logits, item_logits, target_category, target_item, weights=None):
        category_loss = F.cross_entropy(category_logits, target_category, reduction='none')
        item_loss = F.cross_entropy(item_logits, target_item, reduction='none')
        
        if weights is not None:
            category_loss = category_loss * weights
            item_loss = item_loss * weights
            
        category_loss = category_loss.mean()
        item_loss = item_loss.mean()
        
        return item_loss * 0.8 + category_loss * 0.2

def data_processing(data_path, session_length=30):
    events = pd.read_csv(os.path.join(data_path, 'events.csv'))
    categories = pd.read_csv(os.path.join(data_path, 'category_tree.csv'))
    items = pd.concat([pd.read_csv(os.path.join(data_path, 'item_properties_part1.csv')),
                       pd.read_csv(os.path.join(data_path, 'item_properties_part2.csv'))])
    
    item2category = categories.groupby('categoryid')['parentid'].apply(list).to_dict()
    category2items = categories.groupby('parentid')['categoryid'].apply(list).to_dict()

    events = events.rename(columns={
        'visitorid': 'visitor_id',
        'event': 'event_type',
        'timestamp': 'event_time'
    })

    events['category_id'] = events['itemid'].map(item2category)

    items['timestamp'] = pd.to_datetime(items['timestamp'], unit='ms')
    events['event_time'] = pd.to_datetime(events['event_time'], unit='ms')

    items = items.sort_values(['timestamp','itemid'])


    events = events.sort_values(['event_time', 'itemid'])

    events = pd.merge_asof(
        events,
        items,
        left_on='event_time',
        right_on='timestamp',
        by='itemid',
        direction='backward'
    )
    events.dropna(subset=['value'], inplace=True)
    events.rename(columns={'itemid': 'item_id', 'property': 'item_property', 'value': 'item_property_value'}, inplace=True)
    events.drop(columns=['transactionid', 'timestamp'], inplace=True)
    events = events.sort_values(['visitor_id', 'event_time'])
    events['timestamp'] = pd.to_datetime(events['event_time'], unit='ms')
    events['time_diff'] = events.groupby('visitor_id')['timestamp'].diff().dt.total_seconds()
    events['session_id'] = ((events['time_diff'] > session_length*60) | (events['time_diff'].isna())).astype(int).groupby(events['visitor_id']).cumsum()
    events.groupby(['visitor_id', 'session_id'])

    events['event_weight'] = 1.0
    events.loc[events['event_type'] == 'addtocart', 'event_weight'] = 3.0
    events.loc[events['event_type'] == 'transaction', 'event_weight'] = 5.0

    item_properties = items['property'].unique()
    print(events)
    print(events.columns)
    return

    visitor_sequences = []
    item_sequences = []
    category_sequences = []
    item_weights = []
    item_event_types = []
    
    for visitor_id, visitor_df in events.groupby('visitor_id'):
        items = visitor_df['item_id'].tolist()
        categories = visitor_df['category_id'].tolist()
        weights = visitor_df['event_weight'].tolist()
        event_types = visitor_df['event_type'].tolist()
        
        visitor_sequences.append(visitor_id)
        item_sequences.append(items)
        category_sequences.append(categories)
        item_weights.append(weights)
        item_event_types.append(event_types)
    
    item_vocab_size = events['item_id'].max() + 1
    category_vocab_size = events['category_id'].max() + 1
    
    return {
        'visitor_sequences': visitor_sequences,
        'item_sequences': item_sequences,
        'category_sequences': category_sequences,
        'item_weights': item_weights,
        'item_event_types': item_event_types,
        'item_vocab_size': item_vocab_size,
        'category_vocab_size': category_vocab_size
    }

def create_data_loaders(data, batch_size=64):
    num_sequences = len(data['item_sequences'])
    indices = list(range(num_sequences))
    random.shuffle(indices)
    
    train_cutoff = int(0.8 * num_sequences)
    train_indices = indices[:train_cutoff]
    val_indices = indices[train_cutoff:]
    
    train_dataset = SequenceDataset(
        [data['item_sequences'][i] for i in train_indices],
        [data['category_sequences'][i] for i in train_indices],
        [data['item_weights'][i] for i in train_indices],
        [data['item_event_types'][i] for i in train_indices]
    )
    
    val_dataset = SequenceDataset(
        [data['item_sequences'][i] for i in val_indices],
        [data['category_sequences'][i] for i in val_indices],
        [data['item_weights'][i] for i in val_indices],
        [data['item_event_types'][i] for i in val_indices]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset, val_dataset

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=0.001, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            item_seqs = batch['item_seq'].to(device)
            cat_seqs = batch['category_seq'].to(device)
            item_targets = batch['item_target'].to(device)
            category_targets = batch['category_target'].to(device)
            weights = batch['target_weight'].to(device)
            
            optimizer.zero_grad()
            cat_logits, item_logits = model(item_seqs, cat_seqs)
            
            loss = model.calculate_loss(cat_logits, item_logits, category_targets, item_targets, weights)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                item_seqs = batch['item_seq'].to(device)
                cat_seqs = batch['category_seq'].to(device)
                item_targets = batch['item_target'].to(device)
                category_targets = batch['category_target'].to(device)
                
                cat_logits, item_logits = model(item_seqs, cat_seqs)
                loss = model.calculate_loss(cat_logits, item_logits, category_targets, item_targets)
                val_loss += loss.item()
        
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

def evaluate_model(model, dataset, k=10, device='cuda'):
    model.eval()
    
    eval_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    hit_sum = 0
    mrr_sum = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            item_seqs = batch['item_seq'].to(device)
            cat_seqs = batch['category_seq'].to(device)
            item_targets = batch['item_target'].to(device)
            
            _, item_logits = model(item_seqs, cat_seqs)
            _, top_k_items = torch.topk(item_logits, k, dim=1)
            
            for i in range(len(item_targets)):
                target = item_targets[i].item()
                
                if target in top_k_items[i]:
                    hit_sum += 1
                    rank = torch.where(top_k_items[i] == target)[0][0].item() + 1
                    mrr_sum += 1.0 / rank
                
                total_samples += 1
    
    hit_rate = hit_sum / total_samples
    mrr = mrr_sum / total_samples
    
    return {'hit@k': hit_rate, 'mrr': mrr}

def main(data_path="data/", output_dir="models/"):
    # Load and process data
    data = data_processing(data_path)
    return
    
    # Create data loaders
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(data, batch_size=64)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SequenceModel(
        num_items=data['item_vocab_size'],
        num_categories=data['category_vocab_size'],
        embedding_dim=64
    ).to(device)
    
    # Train model
    model = train_model(model, train_loader, val_loader, epochs=3, device=device)
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "recommender_model.pt"))
    
    # Evaluate model
    results = evaluate_model(model, val_dataset, device=device)
    print(f"Final evaluation: Hit@10 = {results['hit@k']:.4f}, MRR = {results['mrr']:.4f}")
    
    return results

if __name__ == "__main__":
    main()