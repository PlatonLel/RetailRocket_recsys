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
        seq_length=seq_length
    )
    
    val_dataset = SequenceDataset(
        item_sequences=[data['item_sequences'][i] for i in val_indices],
        category_sequences=[data['category_sequences'][i] for i in val_indices],
        property_type_sequences=[data['property_type_sequences'][i] for i in val_indices],
        property_value_sequences=[data['property_value_sequences'][i] for i in val_indices],
        seq_length=seq_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset, val_dataset

class DynamicWeightLoss(nn.Module):
    """
    Implements learnable loss weights for multi-task learning.
    Based on the paper "Multi-Task Learning Using Uncertainty to Weigh Losses"
    by Kendall et al.
    """
    def __init__(self, num_tasks=2):
        super(DynamicWeightLoss, self).__init__()
        # Initialize log variances (we work in log space for numerical stability)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses):
        """
        Args:
            losses: List of loss tensors from different tasks
        Returns:
            Total weighted loss
        """
        weights = torch.exp(-self.log_vars)  # Lower variance -> higher weight
        # Apply weights to each loss individually and sum
        weighted_losses = torch.stack([(weights[i] * losses[i] + 0.5 * self.log_vars[i]) for i in range(len(losses))])
        return weighted_losses.sum(), weights

def train_model(model, 
                train_loader, 
                val_loader, 
                epochs=3, 
                learning_rate=0.001, 
                loss_weights={'item': 0.7, 'category': 0.3},
                device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    
    # Initialize dynamic weight loss module
    dynamic_weight = DynamicWeightLoss(num_tasks=2).to(device)
    # Add parameters from dynamic_weight to the optimizer
    optimizer.add_param_group({'params': dynamic_weight.parameters()})
    
    # For tracking performance and adaptive weights
    best_val_loss = float('inf')
    item_weights = []
    category_weights = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        scaler = torch.amp.GradScaler()
        
        epoch_item_weight = 0.0
        epoch_category_weight = 0.0
        num_batches = 0

        for batch in train_loader:
            item_seqs = batch['item_seq'].to(device)
            cat_seqs = batch['category_seq'].to(device)
            prop_type_seqs = batch['prop_type_seq'].to(device)
            prop_value_seqs = batch['prop_value_seq'].to(device)
            item_targets = batch['item_target'].to(device)
            category_targets = batch['category_target'].to(device)

            optimizer.zero_grad()
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                cat_logits, item_logits = model(item_sequences=item_seqs, 
                                                category_sequences=cat_seqs, 
                                                prop_type_sequences=prop_type_seqs, 
                                                prop_value_sequences=prop_value_seqs)

                item_loss = F.cross_entropy(item_logits, item_targets)
                cat_loss = F.cross_entropy(cat_logits, category_targets)
                
                # Use dynamic weighting instead of fixed weights
                loss, weights = dynamic_weight([item_loss, cat_loss])
                
                # Track the learned weights for monitoring
                epoch_item_weight += weights[0].item()
                epoch_category_weight += weights[1].item()
                num_batches += 1
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Track average weights for this epoch
        avg_item_weight = epoch_item_weight / num_batches
        avg_category_weight = epoch_category_weight / num_batches
        item_weights.append(avg_item_weight)
        category_weights.append(avg_category_weight)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_item_loss = 0.0
        val_cat_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                item_seqs = batch['item_seq'].to(device)
                cat_seqs = batch['category_seq'].to(device)
                prop_type_seqs = batch['prop_type_seq'].to(device)
                prop_value_seqs = batch['prop_value_seq'].to(device)
                item_targets = batch['item_target'].to(device)
                category_targets = batch['category_target'].to(device)

                cat_logits, item_logits = model(item_sequences=item_seqs, 
                                                category_sequences=cat_seqs, 
                                                prop_type_sequences=prop_type_seqs, 
                                                prop_value_sequences=prop_value_seqs)
                cat_loss_val = F.cross_entropy(cat_logits, category_targets)
                item_loss_val = F.cross_entropy(item_logits, item_targets)

                # Apply same dynamic weights for validation loss
                losses_val = [item_loss_val, cat_loss_val]
                loss_val, _ = dynamic_weight(losses_val)
                
                val_loss += loss_val.item()
                val_item_loss += item_loss_val.item()
                val_cat_loss += cat_loss_val.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_item_loss = val_item_loss / len(val_loader)
        avg_val_cat_loss = val_cat_loss / len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Dynamic Weights - Item: {avg_item_weight:.4f}, Category: {avg_category_weight:.4f}")
        print(f"Val Losses - Item: {avg_val_item_loss:.4f}, Category: {avg_val_cat_loss:.4f}")
        
        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
    
    # Return the best model based on validation loss
    model.load_state_dict(best_model_state)
    print(f"Final learned weights - Item: {item_weights[-1]:.4f}, Category: {category_weights[-1]:.4f}")
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
            item_targets = batch['item_target'].to(device)

            _, item_logits = model(item_sequences=item_seqs, 
                                    category_sequences=cat_seqs, 
                                    prop_type_sequences=prop_type_seqs, 
                                    prop_value_sequences=prop_value_seqs)
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
    loss_weights={'item': 0.7, 'category': 0.3}):
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
                            epochs=epochs, 
                            device=device, 
                            learning_rate=learning_rate,
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