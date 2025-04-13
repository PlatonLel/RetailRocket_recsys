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
                 event_types_sequences=None,
                 item_weights=None,
                 seq_length=5):
        self.samples = []
        
        for idx in range(len(item_sequences)):
            seq_items = item_sequences[idx]
            seq_cats = category_sequences[idx]
            seq_prop_types = property_type_sequences[idx]
            seq_prop_values = property_value_sequences[idx]
            seq_weights = item_weights[idx] if item_weights else [1.0] * len(seq_items)
            seq_event_types = event_types_sequences[idx] if event_types_sequences else ["view"] * len(seq_items)
            
            for j in range(len(seq_items) - 1):
                item_input = seq_items[:j+1][-seq_length:]
                category_input = seq_cats[:j+1][-seq_length:]
                prop_type_input = seq_prop_types[:j+1][-seq_length:]
                prop_value_input = seq_prop_values[:j+1][-seq_length:]
                event_type_input = seq_event_types[:j+1][-seq_length:]
                padding_size = seq_length - len(item_input)
                if padding_size > 0:
                    item_input = [0] * padding_size + item_input
                    category_input = [0] * padding_size + category_input
                    prop_type_input = [0] * padding_size + prop_type_input
                    prop_value_input = [0] * padding_size + prop_value_input
                    event_type_input = [0] * padding_size + event_type_input
                item_target = seq_items[j+1]
                category_target = seq_cats[j+1]
                target_event_type = seq_event_types[j+1]
                
                sample = {
                    'item_seq': item_input,
                    'category_seq': category_input,
                    'prop_type_seq': prop_type_input,
                    'event_type_seq': event_type_input,
                    'prop_value_seq': prop_value_input, 
                    'item_target': item_target,
                    'category_target': category_target,
                    'target_event_type': target_event_type,
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
            'target_event_type': sample['target_event_type']
        }

class SequenceModel(nn.Module):
    def __init__(self,
                 num_items,
                 num_categories,
                 num_prop_types,
                 num_prop_values,
                 num_event_types,
                 embedding_dim=64,
                 prop_embedding_dim=32,
                 hidden_dim=128): # Hidden dimension for the GRU
        super(SequenceModel, self).__init__()

        self.item_embeddings = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.category_embeddings = nn.Embedding(num_categories, embedding_dim, padding_idx=0)

        self.prop_type_embeddings = nn.Embedding(num_prop_types, prop_embedding_dim, padding_idx=0)
        self.prop_value_embeddings = nn.Embedding(num_prop_values, prop_embedding_dim, padding_idx=0)
        self.event_type_embeddings = nn.Embedding(num_event_types, prop_embedding_dim, padding_idx=0)
        # Calculate combined embedding size
        combined_embedding_dim = (embedding_dim * 2) + (prop_embedding_dim * 3)

        # Single GRU to process the combined sequence
        self.gru = nn.GRU(combined_embedding_dim, hidden_dim, batch_first=True)

        # Predictors using the final GRU hidden state
        self.category_predictor = nn.Linear(hidden_dim, num_categories)
        # The item predictor uses the GRU hidden state. You could potentially
        # concatenate the target category embedding if needed, but let's start simple.
        self.item_predictor = nn.Linear(hidden_dim, num_items) # Predict based on sequence summary
        self.event_type_predictor = nn.Linear(hidden_dim, num_event_types)
    def forward(self,
            item_sequences, 
            category_sequences, 
            prop_type_sequences, 
            prop_value_sequences,
            event_type_sequences):

        ''' batch_size = item_sequences.size(0) '''
        ''' seq_length = item_sequences.size(1) # Get sequence length '''

        # Get embeddings for all features
        item_emb = self.item_embeddings(item_sequences)          # Shape: [batch_size, seq_length, embedding_dim]
        category_emb = self.category_embeddings(category_sequences)
        prop_type_emb = self.prop_type_embeddings(prop_type_sequences) # Shape: [batch_size, seq_length, embedding_dim]
        prop_value_emb = self.prop_value_embeddings(prop_value_sequences) # Shape: [batch_size, seq_length, embedding_dim]
        event_type_emb = self.event_type_embeddings(event_type_sequences) # Shape: [batch_size, seq_length, embedding_dim]
        # Concatenate embeddings along the feature dimension
        combined_emb = torch.cat([item_emb, category_emb, prop_type_emb, prop_value_emb, event_type_emb], dim=2)

        gru_output, final_hidden_state = self.gru(combined_emb)
        final_hidden_state = final_hidden_state.squeeze(0)

        # Predict categories and items based on the final hidden state
        category_logits = self.category_predictor(final_hidden_state)
        item_logits = self.item_predictor(final_hidden_state)
        event_type_logits = self.event_type_predictor(final_hidden_state)

        return category_logits, item_logits, event_type_logits

    def calculate_loss(self, 
                    category_logits, 
                    item_logits, 
                    target_category, 
                    target_item, 
                    target_event_type, 
                    event_type_logits, 
                    weights=None):
        category_loss = F.cross_entropy(category_logits, target_category, reduction='none')
        item_loss = F.cross_entropy(item_logits, target_item, reduction='none')
        event_type_loss = F.cross_entropy(event_type_logits, target_event_type, reduction='none')

        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
                
            category_loss = category_loss * weights.squeeze() # Squeeze back if necessary
            item_loss = item_loss * weights.squeeze()
            event_type_loss = event_type_loss * weights.squeeze()

        category_loss = category_loss.mean()
        item_loss = item_loss.mean()
        event_type_loss = event_type_loss.mean()
        # Adjust weighting between item and category loss if desired
        return item_loss * 0.6 + category_loss * 0.2 + event_type_loss * 0.2

def data_processing(data_path, session_length=30):
    print("Loading data...")
    events = pd.read_csv(os.path.join(data_path, 'events.csv'))
    categories = pd.read_csv(os.path.join(data_path, 'category_tree.csv'))
    items_part1 = pd.read_csv(os.path.join(data_path, 'item_properties_part1.csv'))
    items_part2 = pd.read_csv(os.path.join(data_path, 'item_properties_part2.csv'))
    items = pd.concat([items_part1, items_part2], ignore_index=True)
    del items_part1, items_part2

    # --- Prepare Categories ---
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

    # --- Prepare Core Events ---
    print("Preparing core events...")
    events = events.rename(columns={
        'visitorid': 'visitor_id',
        'event': 'event_type',
        'timestamp': 'event_time'
    })
    events['event_time'] = pd.to_datetime(events['event_time'], unit='ms')
    if 'transactionid' in events.columns:
        events.drop(columns=['transactionid'], inplace=True)

    # Define unique event by visitor, item, and time.
    event_core_cols = ['visitor_id', 'itemid', 'event_time', 'event_type']
    core_events = events[event_core_cols].drop_duplicates().copy()
    core_events = core_events.sort_values(['visitor_id', 'event_time']) # Essential for session logic

    # --- Prepare Item Properties for Merge ---
    print("Preparing item properties...")
    items = items.rename(columns={'timestamp': 'property_time', 'value': 'property_value'})
    items['property_time'] = pd.to_datetime(items['property_time'], unit='ms')
    # IMPORTANT: Assume 'property' and 'property_value' are already numerically encoded here
    items = items.sort_values(['itemid', 'property_time'])

    # --- Merge Properties Temporally ---
    print("Merging properties with events temporally...")
    events_with_properties = pd.merge_asof(
        core_events.sort_values('event_time'),
        items[['itemid', 'property_time', 'property', 'property_value']].sort_values('property_time'), # Select only needed columns
        left_on='event_time',
        right_on='property_time',
        by='itemid',
        direction='backward'
    )
    # Drop rows where no property could be matched backward in time
    events_with_properties.dropna(subset=['property_time'], inplace=True)

    # --- Deduplicate based on Core Event, keeping first property match ---
    print("Selecting first property match per event...")
    # Sort by event time primarily, maybe property time secondarily if needed
    events_with_properties = events_with_properties.sort_values(['visitor_id', 'event_time', 'property_time'])
    # Keep only the first row for each unique event
    final_events = events_with_properties.drop_duplicates(subset=event_core_cols, keep='first').copy()
    # Now final_events has one row per core event, with 'property' and 'property_value' columns
    # representing the first matched property from that time period.

    # --- Sessionization (on Final Events) ---
    print("Performing sessionization...")
    # Calculate diff on the deduplicated data
    final_events['time_diff'] = final_events.groupby('visitor_id')['event_time'].diff().dt.total_seconds()
    final_events['session_id'] = ((final_events['time_diff'] > session_length*60) | (final_events['time_diff'].isna())).astype(int).groupby(final_events['visitor_id']).cumsum()

    # --- Add Category and Weights ---
    print("Adding category and weights...")
    final_events['category_id'] = final_events['itemid'].map(item_to_category_map)
    final_events['category_id'] = final_events['category_id'].fillna(0).astype(int)

    final_events['event_weight'] = 1.0
    final_events.loc[final_events['event_type'] == 'addtocart', 'event_weight'] = 3.0
    final_events.loc[final_events['event_type'] == 'transaction', 'event_weight'] = 5.0
    event_type_vocab_size = 4

    event_type_map = {'view': 1, 'addtocart': 2, 'transaction': 3}
    final_events['event_type_encoded'] = final_events['event_type'].map(event_type_map).fillna(0).astype(int)



    # --- Prepare Sequences ---
    print("Preparing sequences...")
    # Group the final, deduplicated event data
    session_grouped_events = final_events.sort_values('event_time').groupby(['visitor_id', 'session_id'])

    item_sequences = []
    category_sequences = []

    property_type_sequences_orig = []
    property_value_sequences_str = [] # Store strings first
    item_weights = []
    event_types_sequences = []

    for (visitor_id, session_id), session_df in session_grouped_events:
        items_in_session = session_df['itemid'].tolist()
        if len(items_in_session) > 1:
            item_sequences.append(items_in_session)
            category_sequences.append(session_df['category_id'].tolist())
            # Store original property type, fill NaN with a distinct value like -1 for now
            property_type_sequences_orig.append(session_df['property'].fillna(-1).astype(int).tolist())
            # Store property value strings, fill NaNs with a placeholder like 'PAD'
            property_value_sequences_str.append(session_df['property_value'].fillna('PAD').astype(str).tolist())
            item_weights.append(session_df['event_weight'].tolist())
            event_types_sequences.append(session_df['event_type_encoded'].tolist())

    # --- Build Vocabulary for Property Values (remains the same) ---
    print("Building property value vocabulary...")
    # ... (prop_value_map creation and conversion to property_value_sequences_int is correct) ...
    all_prop_values_flat = [val for seq in property_value_sequences_str for val in seq]
    unique_prop_values = sorted(list(set(all_prop_values_flat)))
    prop_value_map = {val: i + 1 for i, val in enumerate(unique_prop_values) if val != 'PAD'}
    prop_value_map['PAD'] = 0
    property_value_sequences_int = []
    for seq_str in property_value_sequences_str:
        property_value_sequences_int.append([prop_value_map.get(val, 0) for val in seq_str])


    # --- Build Vocabulary for Property Types (NEW MAPPING) ---
    print("Building property type vocabulary...")
    # Flatten to find all unique original property type codes (including negatives)
    all_prop_types_flat = [ptype for seq in property_type_sequences_orig for ptype in seq]
    # Get unique original codes, excluding the temporary fillna value (-1) if it wasn't a real code
    unique_prop_types_orig = sorted([p for p in list(set(all_prop_types_flat)) if p != -1])

    # Create mapping: Original Code -> New Contiguous ID (starting from 1)
    prop_type_map = {orig_code: i + 1 for i, orig_code in enumerate(unique_prop_types_orig)}
    # Map the temporary fillna value (-1) to padding index 0
    prop_type_map[-1] = 0

    # Convert original property type sequences to new mapped ID sequences
    property_type_sequences_mapped = []
    for seq_orig in property_type_sequences_orig:
        property_type_sequences_mapped.append([prop_type_map.get(orig_code, 0) for orig_code in seq_orig]) # Default to 0 if unseen


    # --- Calculate Vocab Sizes ---
    print("Calculating vocab sizes...")
    all_items = final_events['itemid'].unique().astype(int)
    all_categories = final_events['category_id'].unique().astype(int)
    # Use the new map size for property type vocab
    prop_type_vocab_size = len(prop_type_map)
    # Property value vocab size calculation remains the same
    prop_value_vocab_size = len(prop_value_map)

    item_vocab_size = int(np.max(all_items)) + 1 if len(all_items) > 0 else 1
    category_vocab_size = int(np.max(all_categories)) + 1 if len(all_categories) > 0 else 1
    # REMOVED old calculation based on np.max(all_prop_types)


    print("Data processing finished.")
    print(f"Property Value Vocab size: {prop_value_vocab_size}")
    print(f"Property Type Vocab size (Mapped): {prop_type_vocab_size}") # Use new size
    print(f"Item vocab size: {item_vocab_size}")
    print(f"Category vocab size: {category_vocab_size}")
    return {
        'item_sequences': item_sequences,
        'category_sequences': category_sequences,
        'property_type_sequences': property_type_sequences_mapped, # Return MAPPED sequences
        'property_value_sequences': property_value_sequences_int,
        'item_weights': item_weights,
        'event_types_sequences': event_types_sequences,
        'event_type_vocab_size': event_type_vocab_size,
        'item_vocab_size': item_vocab_size,
        'category_vocab_size': category_vocab_size,
        'prop_type_vocab_size': prop_type_vocab_size, # Return correct mapped size
        'prop_value_vocab_size': prop_value_vocab_size,
        # 'prop_type_map': prop_type_map # Optional: return map for debugging
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
        event_types_sequences=[data['event_types_sequences'][i] for i in train_indices],
        seq_length=seq_length
    )
    
    val_dataset = SequenceDataset(
        item_sequences=[data['item_sequences'][i] for i in val_indices],
        category_sequences=[data['category_sequences'][i] for i in val_indices],
        property_type_sequences=[data['property_type_sequences'][i] for i in val_indices],
        property_value_sequences=[data['property_value_sequences'][i] for i in val_indices],
        item_weights=[data['item_weights'][i] for i in val_indices],
        event_types_sequences=[data['event_types_sequences'][i] for i in val_indices],
        seq_length=seq_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset, val_dataset

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=0.001, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            item_seqs = batch['item_seq'].to(device)
            cat_seqs = batch['category_seq'].to(device)
            prop_type_seqs = batch['prop_type_seq'].to(device)
            prop_value_seqs = batch['prop_value_seq'].to(device)
            event_type_seqs = batch['event_type_seq'].to(device)
            item_targets = batch['item_target'].to(device)
            category_targets = batch['category_target'].to(device)
            weights = batch['target_weight'].to(device)
            event_type_targets = batch['event_type_target'].to(device)

            optimizer.zero_grad()
            cat_logits, item_logits, event_type_logits = model(item_seqs, cat_seqs, prop_type_seqs, prop_value_seqs, event_type_seqs)
            
            loss = model.calculate_loss(cat_logits, item_logits, category_targets, item_targets, event_type_targets, event_type_logits, weights)
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
                prop_type_seqs = batch['prop_type_seq'].to(device)
                prop_value_seqs = batch['prop_value_seq'].to(device)
                event_type_seqs = batch['event_type_seq'].to(device)
                item_targets = batch['item_target'].to(device)
                category_targets = batch['category_target'].to(device)
                weights = batch['target_weight'].to(device)
                event_type_targets = batch['event_type_target'].to(device)
                cat_logits, item_logits, event_type_logits = model(item_seqs, cat_seqs, prop_type_seqs, prop_value_seqs, event_type_seqs)
                loss = model.calculate_loss(cat_logits, item_logits, category_targets, item_targets, event_type_targets, event_type_logits, weights)
                val_loss += loss.item()
                
        scheduler.step()
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

def evaluate_model(model, dataset, k=10, device='cuda'):
    model.eval()

    eval_loader = DataLoader(dataset, batch_size=64, shuffle=False) # Assuming dataset is SequenceDataset

    # Use defaultdict to store metrics per event type (using INTEGER keys now)
    metrics = defaultdict(lambda: {'hits': 0, 'count': 0, 'mrr_sum': 0.0, 'ndcg_sum': 0.0})
    # Define integer codes for event types we care about (excluding 0 padding)
    event_codes_to_track = [1, 2, 3] # Corresponding to view, addtocart, transaction

    with torch.no_grad():
        for batch in eval_loader:
            item_seqs = batch['item_seq'].to(device)
            cat_seqs = batch['category_seq'].to(device)
            event_type_seqs = batch['event_type_seq'].to(device)
            prop_type_seqs = batch['prop_type_seq'].to(device)
            prop_value_seqs = batch['prop_value_seq'].to(device)
            item_targets = batch['item_target'].to(device)
            # Get the INTEGER event type targets
            target_event_type_codes = batch['event_type_target'].to(device) # This is the integer tensor

            _, item_logits, _ = model(item_seqs, cat_seqs, event_type_seqs, prop_type_seqs, prop_value_seqs)
            _, top_k_items = torch.topk(item_logits, k, dim=1)

            for i in range(len(item_targets)):
                target_item = item_targets[i].item()
                # Get the specific INTEGER event type code for this sample's target
                event_code = target_event_type_codes[i].item() # Get the integer code
                predictions = top_k_items[i].tolist()

                # Ensure we only track defined event codes
                if event_code not in event_codes_to_track:
                    continue

                hit = 0
                rank = float('inf')
                dcg = 0.0 # Use float for dcg

                if target_item in predictions:
                    hit = 1
                    try:
                        rank = predictions.index(target_item) + 1
                        dcg = 1.0 / math.log2(rank + 1) # Use math.log2

                    except ValueError:
                        pass # rank remains inf, dcg remains 0

                mrr = 1.0 / rank if hit else 0.0
                ndcg = dcg

                # Update metrics using the INTEGER event code as the key
                metrics[event_code]['count'] += 1
                metrics[event_code]['hits'] += hit
                metrics[event_code]['mrr_sum'] += mrr
                metrics[event_code]['ndcg_sum'] += ndcg

                # Update overall metrics ('all') - use string key 'all' for consistency
                metrics['all']['count'] += 1
                metrics['all']['hits'] += hit
                metrics['all']['mrr_sum'] += mrr
                metrics['all']['ndcg_sum'] += ndcg

    # --- Calculate final metrics ---
    results = {}
    # Define keys for the results dictionary (use integers for specific types)
    all_keys = ['all'] + event_codes_to_track # Keys are 'all', 1, 2, 3

    # Map codes back to strings for user-friendly output dictionary
    code_to_str_map = {1: 'view', 2: 'addtocart', 3: 'transaction', 'all': 'all'}

    for key in all_keys: # Iterate through 'all', 1, 2, 3
        count = metrics[key]['count']
        output_key = code_to_str_map[key] # Get string key for output ('view', etc.)

        if count > 0:
            hits = metrics[key]['hits']
            mrr_sum = metrics[key]['mrr_sum']
            ndcg_sum = metrics[key]['ndcg_sum']

            hit_rate_at_k = hits / count
            mrr_at_k = mrr_sum / count
            ndcg_at_k = ndcg_sum / count
            precision_at_k = hit_rate_at_k # Correct calculation for P@k: hits / (samples * k)
            recall_at_k = hit_rate_at_k

            # Store results using the STRING key
            results[output_key] = {
                f'hit@{k}': hit_rate_at_k,
                f'mrr@{k}': mrr_at_k,
                f'ndcg@{k}': ndcg_at_k,
                f'precision@{k}': precision_at_k,
                f'recall@{k}': recall_at_k,
                'count': count
            }
        else:
            # Store results using the STRING key
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
    epochs=3, 
    batch_size=64, 
    seq_length=5, 
    learning_rate=0.001,
    session_length=30, 
    k_eval=10):
    # Load and process data
    data = data_processing(data_path, session_length=session_length)
    
    # Create data loaders
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(data,        
                                                                                batch_size=batch_size, 
                                                                                seq_length=seq_length)
    

    # Initialize model
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
            hidden_dim=hidden_dim
        ).to(device)
    except Exception as e:
        print('Error initializing model:', e)
        import traceback
        traceback.print_exc()
        return
    print('Successfully initialized model')
    
    try:    
        # Train model
        model = train_model(model, train_loader, val_loader, epochs=epochs, device=device, learning_rate=learning_rate)
    except Exception as e:
        print('Error training model:', e)
        import traceback
        traceback.print_exc()
        return
    print('Successfully trained model')
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "recommender_model.pt"))
    
    # Evaluate model
    results = evaluate_model(model, val_dataset, k=k_eval, device=device)
    # Print detailed results
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