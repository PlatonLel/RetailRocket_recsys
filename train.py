import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from data_processing import data_processing, prepare_bpr_data
from dataset import BPRDataset
from torch.utils.data import DataLoader
from model import BPRMatrixFactorization
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

def create_bpr_loaders(data, batch_size=1024):
    triplets, user_item_dict, all_items = prepare_bpr_data(
        data['item_sequences'],
        data['category_sequences'],
        data['property_type_sequences'],
        data['property_value_sequences']
    )
    train_cutoff = int(0.8 * len(triplets))
    train_triplets = triplets[:train_cutoff]
    val_triplets = triplets[train_cutoff:]
    train_dataset = BPRDataset(train_triplets)
    val_dataset = BPRDataset(val_triplets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

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
            user_ids = batch['user_id'].to(device)
            pos_item_ids = batch['pos_item_id'].to(device)
            neg_item_ids = batch['neg_item_id'].to(device)
            pos_cat = batch['pos_cat'].to(device)
            neg_cat = batch['neg_cat'].to(device)
            pos_prop_type = batch['pos_prop_type'].to(device)
            neg_prop_type = batch['neg_prop_type'].to(device)
            pos_prop_value = batch['pos_prop_value'].to(device)
            neg_prop_value = batch['neg_prop_value'].to(device)
            diff = model(user_ids, pos_item_ids, neg_item_ids, pos_cat, neg_cat, pos_prop_type, pos_prop_value, neg_prop_type, neg_prop_value)
            loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch['user_id'].to(device)
                pos_item_ids = batch['pos_item_id'].to(device)
                pos_cat = batch['pos_cat'].to(device)
                pos_prop_type = batch['pos_prop_type'].to(device)
                pos_prop_value = batch['pos_prop_value'].to(device)
                pos_score = model(user_ids, pos_item_ids, None, pos_cat, None, pos_prop_type, pos_prop_value, None, None)
                loss = -torch.log(torch.sigmoid(pos_score) + 1e-10).mean()
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    return model

def evaluate_model(model, val_loader, k=10, device='cuda', num_negatives=100):
    model.eval()
    metrics = {'hit': 0, 'mrr': 0, 'ndcg': 0, 'count': 0}
    all_items = torch.arange(model.item_embeddings.num_embeddings, device=device)
    rng = np.random.default_rng()
    with torch.no_grad():
        for batch in val_loader:
            user_ids = batch['user_id'].to(device)
            pos_item_ids = batch['pos_item_id'].to(device)
            pos_cat = batch['pos_cat'].to(device)
            pos_prop_type = batch['pos_prop_type'].to(device)
            pos_prop_value = batch['pos_prop_value'].to(device)
            batch_size = user_ids.size(0)
            for i in range(batch_size):
                user = user_ids[i].unsqueeze(0)
                pos_item = pos_item_ids[i].unsqueeze(0)
                cat = pos_cat[i].unsqueeze(0)
                prop_type = pos_prop_type[i].unsqueeze(0)
                prop_value = pos_prop_value[i].unsqueeze(0)
                # Sample negatives (excluding the positive item)
                neg_items = rng.choice(
                    all_items.cpu().numpy(),
                    size=num_negatives,
                    replace=False
                )
                neg_items = [item for item in neg_items if item != pos_item.item()]
                neg_items = torch.tensor(neg_items[:num_negatives], device=device)
                # Repeat user and features for negatives
                user_rep = user.repeat(len(neg_items) + 1)
                cat_rep = cat.repeat(len(neg_items) + 1)
                prop_type_rep = prop_type.repeat(len(neg_items) + 1)
                prop_value_rep = prop_value.repeat(len(neg_items) + 1)
                # Items: positive + negatives
                items = torch.cat([pos_item, neg_items])
                # Scores
                scores = model(
                    user_rep,
                    items,
                    None,
                    cat_rep,
                    None,
                    prop_type_rep,
                    prop_value_rep,
                    None,
                    None
                )
                # Rank the positive item
                _, indices = torch.topk(scores, k)
                predictions = items[indices].tolist()
                metrics['count'] += 1
                if pos_item.item() in predictions:
                    metrics['hit'] += 1
                    rank = predictions.index(pos_item.item()) + 1
                    metrics['mrr'] += 1.0 / rank
                    metrics['ndcg'] += 1.0 / np.log2(rank + 1)
    if metrics['count'] > 0:
        print(f"Hit@{k}: {metrics['hit']/metrics['count']:.4f}")
        print(f"MRR@{k}: {metrics['mrr']/metrics['count']:.4f}")
        print(f"NDCG@{k}: {metrics['ndcg']/metrics['count']:.4f}")
    else:
        print("No samples evaluated.")
    return metrics

def main(
    data_path="data/", 
    output_dir="models/", 
    embedding_dim=64, 
    prop_embedding_dim=32, 
    dropout_rate=0.1,
    epochs=5, 
    batch_size=1024, 
    learning_rate=0.001,
    session_length=30):
    data = data_processing(data_path, session_length=session_length)
    train_loader, val_loader = create_bpr_loaders(data, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BPRMatrixFactorization(
        num_users=data['user_vocab_size'],
        num_items=data['item_vocab_size'],
        num_prop_types=data['prop_type_vocab_size'],
        num_prop_values=data['prop_value_vocab_size'],
        num_categories=data['category_vocab_size'],
        embedding_dim=embedding_dim,
        prop_embedding_dim=prop_embedding_dim,
        dropout_rate=dropout_rate
    ).to(device)
    print('Successfully initialized model')
    # model = train_model(model, train_loader, val_loader, epochs=epochs, device=device, learning_rate=learning_rate)
    # print('Successfully trained model')
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "recommender_model.pt"))
    # Evaluate after training
    print('Evaluating model on validation set:')
    evaluate_model(model, val_loader, k=10, device=device, num_negatives=100)
    return

if __name__ == "__main__":
    main()