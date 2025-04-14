import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import torch.nn.functional as F
import random
from collections import defaultdict
import math
from data_processing import data_processing
from dataset import SequenceDataset
from torch.utils.data import DataLoader
from model import SequenceModel

pd.set_option('future.no_silent_downcasting', True)

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
                loss_weights={'item': 0.7, 'category': 0.3},
                device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            item_seqs = batch['item_seq'].to(device)
            cat_seqs = batch['category_seq'].to(device)
            prop_type_seqs = batch['prop_type_seq'].to(device)
            prop_value_seqs = batch['prop_value_seq'].to(device)
            event_type_seqs = batch['event_type_seq'].to(device)
            item_targets = batch['item_target'].to(device)
            category_targets = batch['category_target'].to(device)

            optimizer.zero_grad()
            cat_logits, item_logits = model(item_sequences=item_seqs, 
                                            category_sequences=cat_seqs, 
                                            prop_type_sequences=prop_type_seqs, 
                                            prop_value_sequences=prop_value_seqs,
                                            event_type_sequences=event_type_seqs)

            item_loss = F.cross_entropy(item_logits, item_targets)
            cat_loss = F.cross_entropy(cat_logits, category_targets)
            
            loss = (loss_weights['item'] * item_loss + 
                    loss_weights['category'] * cat_loss)
  
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
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

                cat_logits, item_logits = model(item_sequences=item_seqs, 
                                                category_sequences=cat_seqs, 
                                                prop_type_sequences=prop_type_seqs, 
                                                prop_value_sequences=prop_value_seqs,
                                                event_type_sequences=event_type_seqs)
                cat_loss_val = F.cross_entropy(cat_logits, category_targets)
                item_loss_val = F.cross_entropy(item_logits, item_targets)

                loss_val = (loss_weights['item'] * item_loss_val + 
                            loss_weights['category'] * cat_loss_val)
                val_loss += loss_val.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model

def evaluate_model(model, dataset, k=10, device='cuda'):
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=64, shuffle=False) 
    metrics = defaultdict(lambda: {'hits': 0, 'count': 0, 'mrr_sum': 0.0, 'ndcg_sum': 0.0})

    with torch.no_grad():
        for batch in eval_loader:
            item_seqs = batch['item_seq'].to(device)
            cat_seqs = batch['category_seq'].to(device)
            prop_type_seqs = batch['prop_type_seq'].to(device)
            prop_value_seqs = batch['prop_value_seq'].to(device)
            event_type_seqs = batch['event_type_seq'].to(device)
            item_targets = batch['item_target'].to(device)

            _, item_logits = model(item_sequences=item_seqs, 
                                      category_sequences=cat_seqs, 
                                      prop_type_sequences=prop_type_seqs, 
                                      prop_value_sequences=prop_value_seqs,
                                      event_type_sequences=event_type_seqs)
            _, top_k_items = torch.topk(item_logits, k, dim=1)

            for i in range(len(item_targets)):
                target_item = item_targets[i].item()
                predictions = top_k_items[i].tolist()

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
                metrics['all']['count'] += 1
                metrics['all']['hits'] += hit
                metrics['all']['mrr_sum'] += mrr
                metrics['all']['ndcg_sum'] += ndcg

    results = {}
    all_keys = ['all'] 

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
    batch_size=32, 
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