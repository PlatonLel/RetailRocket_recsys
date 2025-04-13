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
import math
pd.set_option('future.no_silent_downcasting', True)

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
                target_event_type_int = seq_event_types_int[j+1]
                target_event_type_str = seq_event_types_str[j+1]
                
                sample = {
                    'item_seq': item_input,
                    'category_seq': category_input,
                    'prop_type_seq': prop_type_input,
                    'event_type_seq': event_type_input,
                    'prop_value_seq': prop_value_input, 
                    'item_target': item_target,
                    'category_target': category_target,
                    'target_event_type': target_event_type_int,
                    'target_event_type_str': target_event_type_str,
                    'target_weight': seq_weights[j+1]
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
            'event_type_target': torch.tensor(sample['target_event_type'], dtype=torch.long),
            'prop_type_seq': torch.tensor(sample['prop_type_seq'], dtype=torch.long),
            'prop_value_seq': torch.tensor(sample['prop_value_seq'], dtype=torch.long),
            'target_weight': torch.tensor(sample['target_weight'], dtype=torch.float),
            'target_event_type_str': sample['target_event_type_str']
        }

class SequenceModel(nn.Module):
    def __init__(self,
                 num_items=None,
                 num_categories=None,
                 num_prop_types=None,
                 num_prop_values=None,
                 num_event_types=None,
                 embedding_dim=64,
                 prop_embedding_dim=32,
                 hidden_dim=128,
                 dropout_rate=0.1):
        super(SequenceModel, self).__init__()

        self.item_embeddings = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.category_embeddings = nn.Embedding(num_categories, embedding_dim, padding_idx=0)

        self.prop_type_embeddings = nn.Embedding(num_prop_types, prop_embedding_dim, padding_idx=0)
        self.prop_value_embeddings = nn.Embedding(num_prop_values, prop_embedding_dim, padding_idx=0)
        self.event_type_embeddings = nn.Embedding(num_event_types, prop_embedding_dim, padding_idx=0)

        combined_embedding_dim = (embedding_dim * 2) + (prop_embedding_dim * 3)

        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(combined_embedding_dim, hidden_dim, batch_first=True)
        self.attention_linear = nn.Linear(hidden_dim, 1)
        self.final_dropout = nn.Dropout(dropout_rate)

        self.category_predictor = nn.Linear(hidden_dim, num_categories)
        self.item_predictor = nn.Linear(hidden_dim, num_items)
        self.event_type_predictor = nn.Linear(hidden_dim, num_event_types)
    def forward(self,
            item_sequences=None, 
            category_sequences=None, 
            prop_type_sequences=None, 
            prop_value_sequences=None,
            event_type_sequences=None):

        # --- Debugging Checks ---
        assert item_sequences.min() >= 0, f"Negative index found in item_sequences: {item_sequences.min()}"
        assert item_sequences.max() < self.item_embeddings.num_embeddings, f"Index >= vocab size in item_sequences. Max: {item_sequences.max()}, Vocab: {self.item_embeddings.num_embeddings}"
        item_emb = self.item_embeddings(item_sequences)

        assert category_sequences.min() >= 0, f"Negative index found in category_sequences: {category_sequences.min()}"
        assert category_sequences.max() < self.category_embeddings.num_embeddings, f"Index >= vocab size in category_sequences. Max: {category_sequences.max()}, Vocab: {self.category_embeddings.num_embeddings}"
        category_emb = self.category_embeddings(category_sequences)

        assert prop_type_sequences.min() >= 0, f"Negative index found in prop_type_sequences: {prop_type_sequences.min()}"
        assert prop_type_sequences.max() < self.prop_type_embeddings.num_embeddings, f"Index >= vocab size in prop_type_sequences. Max: {prop_type_sequences.max()}, Vocab: {self.prop_type_embeddings.num_embeddings}"
        prop_type_emb = self.prop_type_embeddings(prop_type_sequences)

        assert prop_value_sequences.min() >= 0, f"Negative index found in prop_value_sequences: {prop_value_sequences.min()}"
        assert prop_value_sequences.max() < self.prop_value_embeddings.num_embeddings, f"Index >= vocab size in prop_value_sequences. Max: {prop_value_sequences.max()}, Vocab: {self.prop_value_embeddings.num_embeddings}"
        prop_value_emb = self.prop_value_embeddings(prop_value_sequences)

        assert event_type_sequences.min() >= 0, f"Negative index found in event_type_sequences: {event_type_sequences.min()}"
        assert event_type_sequences.max() < self.event_type_embeddings.num_embeddings, f"Index >= vocab size in event_type_sequences. Max: {event_type_sequences.max()}, Vocab: {self.event_type_embeddings.num_embeddings}"
        event_type_emb = self.event_type_embeddings(event_type_sequences)
        # --- End Debugging Checks ---

        combined_emb = torch.cat([item_emb, category_emb, prop_type_emb, prop_value_emb, event_type_emb], dim=2)
        combined_emb = self.embedding_dropout(combined_emb)

        gru_output, final_hidden_state = self.gru(combined_emb)
        attn_scores = self.attention_linear(gru_output)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * gru_output, dim=1)
        context_vector = self.final_dropout(context_vector)

        category_logits = self.category_predictor(context_vector)
        item_logits = self.item_predictor(context_vector)
        event_type_logits = self.event_type_predictor(context_vector)

        return category_logits, item_logits, event_type_logits

    def calculate_loss(self, 
                    category_logits=None, 
                    target_category=None, 
                    target_event_type=None, 
                    event_type_logits=None, 
                    weights=None):
        category_loss = F.cross_entropy(category_logits, target_category, reduction='none')
        event_type_loss = F.cross_entropy(event_type_logits, target_event_type, reduction='none')

        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
                
            category_loss = category_loss * weights.squeeze() 
            event_type_loss = event_type_loss * weights.squeeze()

        category_loss = category_loss.mean()
        event_type_loss = event_type_loss.mean()
        return category_loss, event_type_loss

def data_processing(data_path, session_length=30):
    print("Loading data...")
    events = pd.read_csv(os.path.join(data_path, 'events.csv'))
    categories = pd.read_csv(os.path.join(data_path, 'category_tree.csv'))
    items_part1 = pd.read_csv(os.path.join(data_path, 'item_properties_part1.csv'))
    items_part2 = pd.read_csv(os.path.join(data_path, 'item_properties_part2.csv'))
    items = pd.concat([items_part1, items_part2], ignore_index=True)
    del items_part1, items_part2

    print("Preparing categories...")
    categories = categories.rename(columns={'categoryid': 'category_id', 'parentid': 'parent_id'})
    items.loc[items['property'] == 'available', 'property'] = 1200
    items.loc[items['property'] == 'categoryid', 'property'] = -100
    item_cat_prop = items[items['property'] == -100][['itemid', 'value']].copy()
    item_cat_prop = item_cat_prop.rename(columns={'value': 'category_id'})
    item_cat_prop['category_id'] = pd.to_numeric(item_cat_prop['category_id'], errors='coerce')
    item_cat_prop.dropna(subset=['category_id'], inplace=True)
    item_cat_prop['category_id'] = item_cat_prop['category_id'].astype(int)
    item_to_category_map = item_cat_prop.groupby('itemid')['category_id'].first().to_dict()

    print("Preparing core events...")
    events = events.rename(columns={
        'visitorid': 'visitor_id',
        'event': 'event_type',
        'timestamp': 'event_time'
    })
    events['event_time'] = pd.to_datetime(events['event_time'], unit='ms')
    if 'transactionid' in events.columns:
        events.drop(columns=['transactionid'], inplace=True)

    event_core_cols = ['visitor_id', 'itemid', 'event_time', 'event_type']
    core_events = events[event_core_cols].drop_duplicates().copy()
    core_events = core_events.sort_values(['visitor_id', 'event_time']) # Essential for session logic

    print("Preparing item properties...")
    items = items.rename(columns={'timestamp': 'property_time', 'value': 'property_value'})
    items['property_time'] = pd.to_datetime(items['property_time'], unit='ms')
    items = items.sort_values(['itemid', 'property_time'])

    print("Merging properties with events temporally...")
    events_with_properties = pd.merge_asof(
        core_events.sort_values('event_time'),
        items[['itemid', 'property_time', 'property', 'property_value']].sort_values('property_time'), # Select only needed columns
        left_on='event_time',
        right_on='property_time',
        by='itemid',
        direction='backward'
    )
    events_with_properties.dropna(subset=['property_time'], inplace=True)

    print("Selecting first property match per event...")
    events_with_properties = events_with_properties.sort_values(['visitor_id', 'event_time', 'property_time'])
    final_events = events_with_properties.drop_duplicates(subset=event_core_cols, keep='first').copy()

    print("Performing sessionization...")
    final_events['time_diff'] = final_events.groupby('visitor_id')['event_time'].diff().dt.total_seconds()
    final_events['session_id'] = ((final_events['time_diff'] > session_length*60) | (final_events['time_diff'].isna())).astype(int).groupby(final_events['visitor_id']).cumsum()

    print("Adding category and weights...")
    final_events['category_id'] = final_events['itemid'].map(item_to_category_map)
    final_events['category_id'] = final_events['category_id'].fillna(0).astype(int)

    final_events['event_weight'] = 1.0
    final_events.loc[final_events['event_type'] == 'addtocart', 'event_weight'] = 3.0
    final_events.loc[final_events['event_type'] == 'transaction', 'event_weight'] = 5.0

    print("Preparing sequences...")
    session_grouped_events = final_events.sort_values('event_time').groupby(['visitor_id', 'session_id'])

    item_sequences = []
    category_sequences = []

    property_type_sequences_orig = []
    property_value_sequences_str = []
    item_weights = []
    event_types_sequences = []

    for (visitor_id, session_id), session_df in session_grouped_events:
        items_in_session = session_df['itemid'].tolist()
        if len(items_in_session) > 1:
            item_sequences.append(items_in_session)
            category_sequences.append(session_df['category_id'].tolist())
            property_type_sequences_orig.append(session_df['property'].fillna(-1).astype(int).tolist())
            property_value_sequences_str.append(session_df['property_value'].fillna('PAD').astype(str).tolist())
            item_weights.append(session_df['event_weight'].tolist())
            event_types_sequences.append(session_df['event_type'].tolist())

    print("Building property value vocabulary...")
    all_prop_values_flat = [val for seq in property_value_sequences_str for val in seq]
    unique_prop_values = sorted(list(set(all_prop_values_flat)))
    prop_value_map = {val: i + 1 for i, val in enumerate(unique_prop_values) if val != 'PAD'}
    prop_value_map['PAD'] = 0
    property_value_sequences_int = []
    for seq_str in property_value_sequences_str:
        property_value_sequences_int.append([prop_value_map.get(val, 0) for val in seq_str])


    print("Building property type vocabulary...")
    all_prop_types_flat = [ptype for seq in property_type_sequences_orig for ptype in seq]
    unique_prop_types_orig = sorted([p for p in list(set(all_prop_types_flat)) if p != -1])

    prop_type_map = {orig_code: i + 1 for i, orig_code in enumerate(unique_prop_types_orig)}
    prop_type_map[-1] = 0

    property_type_sequences_mapped = []
    for seq_orig in property_type_sequences_orig:
        property_type_sequences_mapped.append([prop_type_map.get(orig_code, 0) for orig_code in seq_orig])

    print("Calculating vocab sizes...")
    all_items = final_events['itemid'].unique().astype(int)
    all_categories = final_events['category_id'].unique().astype(int)
    prop_type_vocab_size = len(prop_type_map)
    prop_value_vocab_size = len(prop_value_map)

    item_vocab_size = int(np.max(all_items)) + 1 if len(all_items) > 0 else 1
    category_vocab_size = int(np.max(all_categories)) + 1 if len(all_categories) > 0 else 1
    event_type_map_enc = {'view': 1, 'addtocart': 2, 'transaction': 3}
    event_type_vocab_size = 4 
    event_types_sequences_int = []
    for seq_str in event_types_sequences:
        event_types_sequences_int.append([event_type_map_enc.get(etype, 0) for etype in seq_str])


    print("Data processing finished.")
    print(f"Property Value Vocab size: {prop_value_vocab_size}")
    print(f"Property Type Vocab size (Mapped): {prop_type_vocab_size}")
    print(f"Item vocab size: {item_vocab_size}")
    print(f"Category vocab size: {category_vocab_size}")
    print(f"Event type vocab size: {event_type_vocab_size}")
    return {
        'item_sequences': item_sequences,
        'category_sequences': category_sequences,
        'property_type_sequences': property_type_sequences_mapped,
        'property_value_sequences': property_value_sequences_int,
        'item_weights': item_weights,
        'event_types_sequences_int': event_types_sequences_int,
        'event_types_sequences_str': event_types_sequences,
        'event_type_vocab_size': event_type_vocab_size,
        'item_vocab_size': item_vocab_size,
        'category_vocab_size': category_vocab_size,
        'prop_type_vocab_size': prop_type_vocab_size,
        'prop_value_vocab_size': prop_value_vocab_size,
    }


def create_data_loaders(data, batch_size=32, seq_length=5):
    num_sequences = len(data['item_sequences'])
    indices = list(range(num_sequences))
    random.shuffle(indices)
    
    train_cutoff = int(0.8 * num_sequences)
    train_indices = indices[:train_cutoff]
    val_indices = indices[train_cutoff:]
    
    train_dataset = SequenceDataset(
        item_sequences=[data['item_sequences'][i] for i in train_indices],
        category_sequences=[data['category_sequences'][i] for i in train_indices],
        property_type_sequences=[data['property_type_sequences'][i] for i in train_indices],
        property_value_sequences=[data['property_value_sequences'][i] for i in train_indices],
        item_weights=[data['item_weights'][i] for i in train_indices],
        event_types_sequences_int=[data['event_types_sequences_int'][i] for i in train_indices],
        event_types_sequences_str=[data['event_types_sequences_str'][i] for i in train_indices],
        seq_length=seq_length
    )
    
    val_dataset = SequenceDataset(
        item_sequences=[data['item_sequences'][i] for i in val_indices],
        category_sequences=[data['category_sequences'][i] for i in val_indices],
        property_type_sequences=[data['property_type_sequences'][i] for i in val_indices],
        property_value_sequences=[data['property_value_sequences'][i] for i in val_indices],
        item_weights=[data['item_weights'][i] for i in val_indices],
        event_types_sequences_int=[data['event_types_sequences_int'][i] for i in val_indices],
        event_types_sequences_str=[data['event_types_sequences_str'][i] for i in val_indices],
        seq_length=seq_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset, val_dataset

def train_model(model, 
                train_loader, 
                val_loader, 
                num_items,
                epochs=3, 
                learning_rate=0.001, 
                num_negative_samples=4,
                loss_weights={'item': 0.6, 'category': 0.2, 'event_type': 0.2},
                device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    item_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_item_loss = 0.0
        total_cat_loss = 0.0
        total_event_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            item_seqs = batch['item_seq'].to(device)
            cat_seqs = batch['category_seq'].to(device)
            prop_type_seqs = batch['prop_type_seq'].to(device)
            prop_value_seqs = batch['prop_value_seq'].to(device)
            event_type_seqs = batch['event_type_seq'].to(device)

            item_targets = batch['item_target'].to(device)
            category_targets = batch['category_target'].to(device)
            weights = batch['target_weight'].to(device)
            event_type_targets = batch['target_event_type'].to(device)

            optimizer.zero_grad()
            cat_logits, item_logits, event_type_logits = model(item_sequences=None, 
                                                            category_sequences=cat_seqs, 
                                                            prop_type_sequences=prop_type_seqs, 
                                                            prop_value_sequences=prop_value_seqs,
                                                            event_type_sequences=event_type_seqs)

            batch_size = item_targets.size(0)
            neg_item_samples = []
            for i in range(batch_size):
                positive_item = item_targets[i].item()
                negatives = []
                while len(negatives) < num_negative_samples:
                    neg_id = random.randint(1, num_items - 1)
                    if neg_id != positive_item:
                        negatives.append(neg_id)
                neg_item_samples.append(negatives)
            neg_item_samples = torch.tensor(neg_item_samples, dtype=torch.long).to(device)

            pos_item_logits = item_logits.gather(1, item_targets.unsqueeze(1))
            neg_item_logits = item_logits.gather(1, neg_item_samples)
            
            item_logits_sampled = torch.cat([pos_item_logits, neg_item_logits], dim=1)
            pos_targets = torch.ones(batch_size, 1).to(device)
            neg_targets = torch.zeros(batch_size, num_negative_samples).to(device)
            item_targets_binary = torch.cat([pos_targets, neg_targets], dim=1)

            item_loss = item_criterion(item_logits_sampled, item_targets_binary)
            cat_loss, event_loss = model.calculate_loss(category_logits=cat_logits, 
                                                        target_category=category_targets, 
                                                        target_event_type=event_type_targets, 
                                                        event_type_logits=event_type_logits, 
                                                        weights=weights)
            loss = (loss_weights['item'] * item_loss + 
                    loss_weights['category'] * cat_loss + 
                    loss_weights['event_type'] * event_loss)
  
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_item_loss += item_loss.item()
            total_cat_loss += cat_loss.item()
            total_event_loss += event_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_item_loss = total_item_loss / len(train_loader)
        avg_cat_loss = total_cat_loss / len(train_loader)
        avg_event_loss = total_event_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                item_seqs = batch['item_seq'].to(device)
                cat_seqs = batch['category_seq'].to(device)
                prop_type_seqs = batch['prop_type_seq'].to(device)
                prop_value_seqs = batch['prop_value_seq'].to(device)
                event_type_seqs = batch['event_type_seq'].to(device)
                item_targets = batch['item_target'].to(device)
                category_targets = batch['category_target'].to(device)
                weights = batch['target_weight'].to(device)
                event_type_targets = batch['target_event_type'].to(device)

                weights_val = batch['target_weight'].to(device)
                cat_logits, item_logits, event_type_logits = model(item_sequences=item_seqs, 
                                                                    category_sequences=cat_seqs, 
                                                                    prop_type_sequences=prop_type_seqs, 
                                                                    prop_value_sequences=prop_value_seqs, 
                                                                    event_type_sequences=event_type_seqs)
                cat_loss_val, event_loss_val = model.calculate_loss(category_logits=cat_logits, 
                                                                    target_category=category_targets, 
                                                                    target_event_type=event_type_targets, 
                                                                    event_type_logits=event_type_logits, 
                                                                    weights=weights_val)
                item_loss_val = F.cross_entropy(item_logits, item_targets)
                loss_val = (loss_weights['item'] * item_loss_val + 
                            loss_weights['category'] * cat_loss_val + 
                            loss_weights['event_type'] * event_loss_val)
                val_loss += loss_val.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model

def evaluate_model(model, dataset, k=10, device='cuda'):
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=64, shuffle=False) 
    metrics = defaultdict(lambda: {'hits': 0, 'count': 0, 'mrr_sum': 0.0, 'ndcg_sum': 0.0})
    event_codes_to_track = ['view', 'addtocart', 'transaction'] 

    with torch.no_grad():
        for batch in eval_loader:
            item_seqs = batch['item_seq'].to(device)
            cat_seqs = batch['category_seq'].to(device)
            event_type_seqs = batch['event_type_seq'].to(device)
            prop_type_seqs = batch['prop_type_seq'].to(device)
            prop_value_seqs = batch['prop_value_seq'].to(device)
            item_targets = batch['item_target'].to(device)
            target_event_type_codes = batch['target_event_type_str']

            _, item_logits, _ = model(item_sequences=item_seqs, 
                                      category_sequences=cat_seqs, 
                                      prop_type_sequences=prop_type_seqs, 
                                      prop_value_sequences=prop_value_seqs, 
                                      event_type_sequences=event_type_seqs)
            _, top_k_items = torch.topk(item_logits, k, dim=1)

            for i in range(len(item_targets)):
                target_item = item_targets[i].item()
                event_type = target_event_type_codes[i]
                predictions = top_k_items[i].tolist()

                if event_type not in event_codes_to_track:
                    continue

                hit = 0
                rank = float('inf')
                dcg = 0.0 

                if target_item in predictions:
                    hit = 1
                    try:
                        rank = predictions.index(target_item) + 1
                        dcg = 1.0 / math.log2(rank + 1) 

                    except ValueError:
                        pass 

                mrr = 1.0 / rank if hit else 0.0
                ndcg = dcg

                metrics[event_type]['count'] += 1
                metrics[event_type]['hits'] += hit
                metrics[event_type]['mrr_sum'] += mrr
                metrics[event_type]['ndcg_sum'] += ndcg

                metrics['all']['count'] += 1
                metrics['all']['hits'] += hit
                metrics['all']['mrr_sum'] += mrr
                metrics['all']['ndcg_sum'] += ndcg

    results = {}
    all_keys = ['all'] + event_codes_to_track 

    for key in all_keys: 
        count = metrics[key]['count']
        output_key = key

        if count > 0:
            hits = metrics[key]['hits']
            mrr_sum = metrics[key]['mrr_sum']
            ndcg_sum = metrics[key]['ndcg_sum']

            hit_rate_at_k = hits / count
            mrr_at_k = mrr_sum / count
            ndcg_at_k = ndcg_sum / count
            precision_at_k = hit_rate_at_k 
            recall_at_k = hit_rate_at_k

            results[output_key] = {
                f'hit@{k}': hit_rate_at_k,
                f'mrr@{k}': mrr_at_k,
                f'ndcg@{k}': ndcg_at_k,
                f'precision@{k}': precision_at_k,
                f'recall@{k}': recall_at_k,
                'count': count
            }
        else:
            results[output_key] = {
                f'hit@{k}': 0.0, f'mrr@{k}': 0.0, f'ndcg@{k}': 0.0,
                f'precision@{k}': 0.0, f'recall@{k}': 0.0, 'count': 0
            }

    return results


def main(
    data_path="data/", 
    output_dir="models/", 
    embedding_dim=64, 
    prop_embedding_dim=32, 
    hidden_dim=128,
    dropout_rate=0.1,
    epochs=5, 
    batch_size=64, 
    seq_length=10, 
    learning_rate=0.001,
    session_length=30, 
    k_eval=10,
    num_negative_samples=4,
    loss_weights={'item': 0.6, 'category': 0.2, 'event_type': 0.2}):
    data = data_processing(data_path, session_length=session_length)
    
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(data,        
                                                                                batch_size=batch_size, 
                                                                                seq_length=seq_length)    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = SequenceModel(
            num_items=data['item_vocab_size'],
            num_categories=data['category_vocab_size'],
            num_prop_types=data['prop_type_vocab_size'],
            num_prop_values=data['prop_value_vocab_size'],
            num_event_types=data['event_type_vocab_size'],
            embedding_dim=embedding_dim,
            prop_embedding_dim=prop_embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        ).to(device)
    except Exception as e:
        print('Error initializing model:', e)
        import traceback
        traceback.print_exc()
        return
    print('Successfully initialized model')
    
    try:    
        model = train_model(model, 
                            train_loader, 
                            val_loader, 
                            num_items=data['item_vocab_size'],
                            epochs=epochs, 
                            device=device, 
                            learning_rate=learning_rate,
                            num_negative_samples=num_negative_samples,
                            loss_weights=loss_weights)
    except Exception as e:
        print('Error training model:', e)
        import traceback
        traceback.print_exc()
        return
    print('Successfully trained model')
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "recommender_model.pt"))

    results = evaluate_model(model, val_dataset, k=k_eval, device=device)
    for event_type, metrics in results.items():
        print(f"\nMetrics for Event Type: '{event_type}' (Count: {metrics['count']})")
        if metrics['count'] > 0:
            print(f"  Hit@{k_eval}:      {metrics[f'hit@{k_eval}']:.4f}")
            print(f"  MRR@{k_eval}:      {metrics[f'mrr@{k_eval}']:.4f}")
            print(f"  NDCG@{k_eval}:     {metrics[f'ndcg@{k_eval}']:.4f}")
            print(f"  Precision@{k_eval}: {metrics[f'precision@{k_eval}']:.4f}")
            print(f"  Recall@{k_eval}:   {metrics[f'recall@{k_eval}']:.4f}")
        else:
            print("  No samples evaluated.")

    return results

if __name__ == "__main__":
    main()