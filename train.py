import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from data_processing import process_retailrocket_data, build_item_feature_matrix
from dataset import BPRDataset
from torch.utils.data import DataLoader
from model import BPRMatrixFactorization, cbf_recommend

pd.set_option('future.no_silent_downcasting', True)


def train_model(model, 
                train_loader, 
                val_loader, 
                epochs=30, 
                learning_rate=0.001, 
                weight_decay=1e-6,
                device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            user_ids = batch[0].long().to(device)
            pos_item_ids = batch[1].long().to(device)
            neg_item_ids = batch[2].long().to(device)
            weights = batch[3].float().to(device)

            diff = model(user_ids, pos_item_ids, neg_item_ids)
            loss = (-torch.log(torch.sigmoid(diff) + 1e-10) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch[0].long().to(device)
                pos_item_ids = batch[1].long().to(device)
                neg_item_ids = batch[2].long().to(device)
                weights = batch[3].float().to(device)
                diff = model(user_ids, pos_item_ids, neg_item_ids)
                loss = (-torch.log(torch.sigmoid(diff) + 1e-10) * weights).mean()
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    return model

def evaluate_model(model, val_loader, device='cuda', k=10, num_negatives=100):
    model.eval()
    metrics = {'hit': 0, 'mrr': 0, 'ndcg': 0, 'count': 0}
    rng = np.random.default_rng()
    all_items = model.item_embeddings.num_embeddings
    
    fixed_neg_items = rng.integers(0, all_items, size=num_negatives)
    fixed_neg_items_tensor = torch.tensor(fixed_neg_items, device=device).long()
    
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch in val_loader:
            user_ids = batch[0].long().to(device)
            pos_item_ids = batch[1].long().to(device)
            batch_size = user_ids.size(0)
            for i in range(batch_size):
                user = user_ids[i].unsqueeze(0)
                pos_item = pos_item_ids[i].unsqueeze(0)
                
                neg_items = fixed_neg_items_tensor[fixed_neg_items_tensor != pos_item.item()]
                if len(neg_items) < num_negatives:
                    additional_needed = num_negatives - len(neg_items)
                    additional_items = []
                    while len(additional_items) < additional_needed:
                        item = rng.integers(0, all_items)
                        if item != pos_item.item() and item not in neg_items:
                            additional_items.append(item)
                    if additional_items:
                        additional_tensor = torch.tensor(additional_items, device=device).long()
                        neg_items = torch.cat([neg_items, additional_tensor])

                neg_items = neg_items[:num_negatives]
                user_rep = user.repeat(len(neg_items) + 1)
                items = torch.cat([pos_item, neg_items])
                scores = model(user_rep, items)
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend([1] + [0] * len(neg_items))

                _, indices = torch.topk(scores, k)
                predictions = items[indices].tolist()

                metrics['count'] += 1
                if pos_item.item() in predictions:
                    metrics['hit'] += 1
                    rank = predictions.index(pos_item.item()) + 1
                    metrics['mrr'] += 1.0 / rank
                    metrics['ndcg'] += 1.0 / np.log2(rank + 1)

    try:
        roc_auc = roc_auc_score(all_labels, all_scores)
        metrics['roc_auc'] = roc_auc
    except Exception as e:
        print(f"Could not calculate ROC AUC: {e}")
        metrics['roc_auc'] = None
                    
    if metrics['count'] > 0:
        for key in ['hit', 'mrr', 'ndcg']:
            metrics[key] /= metrics['count']
            print(f"{key.upper()}@{k}: {metrics[key]:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    else:
        print("No samples evaluated.")
    return metrics

def main(
    data_path="data/", 
    output_dir="models/", 
    embedding_dim=64, 
    dropout_rate=0.1,
    epochs=5, 
    batch_size=1024, 
    learning_rate=0.001,
    weight_decay=1e-4):
    data = process_retailrocket_data(data_path, 
                                     min_interactions=7, 
                                     weights={'view': 1.0, 
                                              'addtocart': 3.0, 
                                              'transaction': 5.0})
    train_dataset = BPRDataset(data['train_data'], data['num_items'])
    val_dataset = BPRDataset(data['test_data'], data['num_items'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BPRMatrixFactorization(
        num_users=data['num_users'],
        num_items=data['num_items'],
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate
    ).to(device)
    print('Successfully initialized model')
    model = train_model(model, 
                        train_loader, 
                        val_loader, 
                        epochs=epochs, 
                        device=device, 
                        learning_rate=learning_rate, 
                        weight_decay=weight_decay)
    print('Successfully trained model')
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "recommender_model.pt"))

    print('Evaluating model on validation set:')
    evaluate_model(model, val_loader, device=device)
    return

if __name__ == "__main__":
    main(dropout_rate=0.2, epochs=20, weight_decay=1e-2, embedding_dim=64)