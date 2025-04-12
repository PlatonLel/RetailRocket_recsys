"""
Simple Recommender Model for E-commerce

A minimal implementation focused on core functionality to test recommendation concepts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import random
import os
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import time


class SequenceDataset(Dataset):
    """Dataset for sequence recommendation data with weights and negative sampling"""
    def __init__(self, item_sequences, category_sequences, parent_category_sequences=None, 
                 root_category_sequences=None, item_weights=None, seq_length=5, 
                 neg_samples=5, item_vocab_size=None, max_samples_per_sequence=5):
        self.samples = []
        skipped_sequences = 0
        self.item_vocab_size = item_vocab_size
        self.neg_samples = neg_samples
        
        # Store random state for reproducibility
        self.random_state = random.Random(42)
        
        # Handle optional hierarchy data
        has_parent_cats = parent_category_sequences is not None
        has_root_cats = root_category_sequences is not None
        has_weights = item_weights is not None
        
        # For very long sequences, limit the number of samples we extract
        total_potential_samples = sum(max(0, len(seq) - 1) for seq in item_sequences)
        print(f"Potential training samples before limiting: {total_potential_samples}")
        
        # Create training samples from sequences (with sample limiting for speed)
        for i in range(len(item_sequences)):
            seq_items = item_sequences[i]
            seq_cats = category_sequences[i]
            seq_parent_cats = parent_category_sequences[i] if has_parent_cats else None
            seq_root_cats = root_category_sequences[i] if has_root_cats else None
            seq_weights = item_weights[i] if has_weights else [1.0] * len(seq_items)
            
            # Skip sequences with less than 2 items (need at least input + target)
            if len(seq_items) < 2:
                skipped_sequences += 1
                continue
            
            # For long sequences, sample fewer positions to reduce load
            sample_positions = list(range(len(seq_items) - 1))
            if len(sample_positions) > max_samples_per_sequence:
                # Prioritize positions with high-weight targets
                high_weight_positions = [j for j in sample_positions if j < len(seq_weights)-1 and seq_weights[j+1] > 1.0]
                
                # Always include high-weight positions if available (up to half the samples)
                if high_weight_positions:
                    if len(high_weight_positions) > max_samples_per_sequence // 2:
                        high_weight_positions = self.random_state.sample(high_weight_positions, max_samples_per_sequence // 2)
                    
                    # Fill remaining positions randomly
                    remaining_positions = [j for j in sample_positions if j not in high_weight_positions]
                    random_positions = self.random_state.sample(remaining_positions, 
                                                  min(max_samples_per_sequence - len(high_weight_positions), 
                                                      len(remaining_positions)))
                    sample_positions = high_weight_positions + random_positions
                else:
                    # No high-weight positions, just sample randomly
                    sample_positions = self.random_state.sample(sample_positions, max_samples_per_sequence)
            
            # Create samples from selected positions
            for j in sample_positions:
                # Determine effective sequence length based on available history
                actual_seq_length = min(seq_length, j + 1)
                
                # Input sequences (use as much context as available up to seq_length)
                start_pos = max(0, j + 1 - actual_seq_length)
                item_input = seq_items[start_pos:j+1]
                category_input = seq_cats[start_pos:j+1]
                
                # Get parent and root categories if available
                parent_cat_input = seq_parent_cats[start_pos:j+1] if has_parent_cats else None
                root_cat_input = seq_root_cats[start_pos:j+1] if has_root_cats else None
                
                # Get weights for the input items
                weight_input = seq_weights[start_pos:j+1] if has_weights else [1.0] * len(item_input)
                
                # Pad sequences if needed
                if len(item_input) < seq_length:
                    padding = [0] * (seq_length - len(item_input))
                    item_input = padding + item_input
                    category_input = padding + category_input
                    weight_input = [0.0] * len(padding) + weight_input
                    if has_parent_cats:
                        parent_cat_input = padding + parent_cat_input
                    if has_root_cats:
                        root_cat_input = padding + root_cat_input
                
                # Target (next item)
                item_target = seq_items[j+1]
                target_weight = seq_weights[j+1] if has_weights else 1.0
                
                sample = {
                    'item_seq': item_input,
                    'category_seq': category_input,
                    'target': item_target,
                    'weight': weight_input,
                    'target_weight': target_weight
                }
                
                # Add hierarchical category data if available
                if has_parent_cats:
                    sample['parent_category_seq'] = parent_cat_input
                if has_root_cats:
                    sample['root_category_seq'] = root_cat_input
                
                self.samples.append(sample)
        
        print(f"SequenceDataset: Created {len(self.samples)} samples (limited), skipped {skipped_sequences} sequences")
        
        # Generate a fixed set of negative samples for faster training
        if self.item_vocab_size and self.neg_samples > 0:
            self.fixed_neg_samples = {}
            # Don't generate too many different negative samples to save memory
            max_neg_sample_sets = min(1000, len(self.samples))
            for i in range(max_neg_sample_sets):
                self.fixed_neg_samples[i] = self._generate_neg_samples(
                    self.samples[i % len(self.samples)]['target'])

    def _generate_neg_samples(self, target):
        """Generate negative samples for a target (helper method)"""
        neg_samples = []
        while len(neg_samples) < self.neg_samples:
            neg = self.random_state.randint(1, self.item_vocab_size - 1)
            if neg != target and neg not in neg_samples:
                neg_samples.append(neg)
        return neg_samples

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        result = {
            'item_seq': torch.tensor(sample['item_seq'], dtype=torch.long),
            'category_seq': torch.tensor(sample['category_seq'], dtype=torch.long),
            'target': torch.tensor(sample['target'], dtype=torch.long),
            'weight': torch.tensor(sample['weight'], dtype=torch.float),
            'target_weight': torch.tensor(sample['target_weight'], dtype=torch.float)
        }
        
        # Add hierarchical category data if available
        if 'parent_category_seq' in sample:
            result['parent_category_seq'] = torch.tensor(sample['parent_category_seq'], dtype=torch.long)
        if 'root_category_seq' in sample:
            result['root_category_seq'] = torch.tensor(sample['root_category_seq'], dtype=torch.long)
            
        # Generate negative samples if vocab size is available - optimized version
        if self.item_vocab_size and self.neg_samples > 0:
            # Use a faster method with pre-generated candidates
            target = sample['target']
            
            # Use fixed negative sampling pool to speed up processing
            if not hasattr(self, 'neg_pool'):
                # Create a pool of potential negative samples
                self.neg_pool = list(range(1, min(10000, self.item_vocab_size)))
                random.shuffle(self.neg_pool)
            
            # Fast negative sampling
            neg_samples = []
            neg_idx = 0
            while len(neg_samples) < self.neg_samples and neg_idx < len(self.neg_pool):
                neg = self.neg_pool[neg_idx]
                if neg != target and neg not in neg_samples:
                    neg_samples.append(neg)
                neg_idx += 1
                
            # If we didn't get enough samples, add some random ones
            while len(neg_samples) < self.neg_samples:
                neg = random.randint(1, self.item_vocab_size - 1)
                if neg != target and neg not in neg_samples:
                    neg_samples.append(neg)
            
            result['neg_samples'] = torch.tensor(neg_samples, dtype=torch.long)
            
        return result


class SimpleRecommenderModel(nn.Module):
    """
    Enhanced recommendation model combining sequential models with category hierarchy information.
    """
    def __init__(
        self, 
        num_items: int,
        num_categories: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        use_hierarchy: bool = True,
        use_negative_sampling: bool = True
    ):
        super(SimpleRecommenderModel, self).__init__()
        
        # Model configuration
        self.num_items = num_items
        self.num_categories = num_categories
        self.use_negative_sampling = use_negative_sampling
        
        # Embedding layers
        self.item_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.category_embeddings = nn.Embedding(num_categories + 1, embedding_dim, padding_idx=0)
        
        # Hierarchical category embeddings (optional)
        self.use_hierarchy = use_hierarchy
        if use_hierarchy:
            self.parent_category_embeddings = nn.Embedding(num_categories + 1, embedding_dim, padding_idx=0)
            self.root_category_embeddings = nn.Embedding(num_categories + 1, embedding_dim, padding_idx=0)
            combined_input_size = embedding_dim * 4  # Item + 3 levels of categories
        else:
            combined_input_size = embedding_dim * 2  # Only item + direct category
        
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=combined_input_size,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Predictors
        self.category_predictor = nn.Linear(hidden_dim, num_categories)
        self.item_predictor = nn.Linear(hidden_dim, num_items)
        
        # Regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, item_sequences, category_sequences, parent_category_sequences=None, root_category_sequences=None):
        """Enhanced forward pass with category hierarchy"""
        # Get embeddings
        item_embeds = self.item_embeddings(item_sequences)
        category_embeds = self.category_embeddings(category_sequences)
        
        # Combine embeddings based on available data
        if self.use_hierarchy and parent_category_sequences is not None and root_category_sequences is not None:
            # Get hierarchical category embeddings
            parent_category_embeds = self.parent_category_embeddings(parent_category_sequences)
            root_category_embeds = self.root_category_embeddings(root_category_sequences)
            
            # Combine all embeddings
            combined = torch.cat([item_embeds, category_embeds, parent_category_embeds, root_category_embeds], dim=2)
        else:
            # Use only direct category embeddings
            combined = torch.cat([item_embeds, category_embeds], dim=2)
        
        combined = self.dropout(combined)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined)
        final_hidden = lstm_out[:, -1, :]  # Take last output
        final_hidden = self.dropout(final_hidden)
        
        # Make predictions
        category_logits = self.category_predictor(final_hidden)
        item_logits = self.item_predictor(final_hidden)
        
        return category_logits, item_logits
    
    def get_item_embeddings(self, item_ids):
        """Get embeddings for specific items"""
        return self.item_embeddings(item_ids)
    
    def compute_item_scores(self, user_hidden, item_ids=None):
        """
        Compute scores for specific items or all items
        
        Args:
            user_hidden: User hidden state from LSTM
            item_ids: Optional list of item IDs to score, if None score all items
            
        Returns:
            Scores for the specified items
        """
        if item_ids is not None:
            # Get embeddings only for the specified items
            item_embeddings = self.item_embeddings(item_ids)
            return torch.matmul(user_hidden.unsqueeze(1), item_embeddings.transpose(1, 2)).squeeze(1)
        else:
            # Use the full predictor for all items
            return self.item_predictor(user_hidden)
    
    def compute_weighted_bce_loss(self, logits, pos_targets, neg_targets, target_weights=None):
        """
        Compute weighted binary cross entropy loss for positive and negative samples
        
        Args:
            logits: Predicted logits
            pos_targets: Positive target items
            neg_targets: Negative sampled items
            target_weights: Optional weights for positive targets
            
        Returns:
            Weighted BCE loss
        """
        batch_size = logits.size(0)
        
        # Extract scores for positive items
        pos_scores = logits.gather(1, pos_targets.unsqueeze(1)).squeeze()
        
        # Extract scores for negative items
        neg_scores = logits.gather(1, neg_targets)
        
        # Apply sigmoid to convert to probabilities
        pos_probs = torch.sigmoid(pos_scores)
        neg_probs = torch.sigmoid(neg_scores)
        
        # Compute binary cross entropy loss
        pos_loss = -torch.log(pos_probs + 1e-8)
        neg_loss = -torch.log(1 - neg_probs + 1e-8).mean(dim=1)
        
        # Apply weights if provided
        if target_weights is not None:
            pos_loss = pos_loss * target_weights
        
        # Combine positive and negative losses
        loss = (pos_loss + neg_loss).mean()
        
        return loss


def evaluate_recommendations(model, item_seq, category_seq, actual_next, parent_category_seq=None, root_category_seq=None, top_k=10):
    """Enhanced evaluation function with support for category hierarchy"""
    model.eval()
    device = next(model.parameters()).device
    
    # Convert to tensors
    item_tensor = torch.tensor([item_seq], dtype=torch.long).to(device)
    category_tensor = torch.tensor([category_seq], dtype=torch.long).to(device)
    
    # Convert hierarchical category data if available
    parent_category_tensor = None
    if parent_category_seq is not None:
        parent_category_tensor = torch.tensor([parent_category_seq], dtype=torch.long).to(device)
        
    root_category_tensor = None
    if root_category_seq is not None:
        root_category_tensor = torch.tensor([root_category_seq], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Get predictions
        _, item_logits = model(item_tensor, category_tensor, parent_category_tensor, root_category_tensor)
        
        # Get top-k items
        _, top_items = torch.topk(item_logits, top_k, dim=1)
        top_items = top_items.squeeze().cpu().numpy().tolist()
    
    # Calculate metrics
    hit = int(actual_next in top_items)
    reciprocal_rank = 1.0 / (top_items.index(actual_next) + 1) if hit else 0.0
    
    return {
        'hit@k': hit,
        'mrr': reciprocal_rank,
        'top_predictions': top_items
    }

def train_simple_model(data, epochs=5, batch_size=32, seq_length=5, learning_rate=0.001, val_ratio=0.1, 
                   use_hierarchy=True, use_negative_sampling=True, neg_samples=5, 
                   max_train_time_minutes=30):  # Add timeout parameter
    """Train the model on provided data with enhanced category hierarchy and negative sampling"""
    start_time = time.time()
    max_train_time_seconds = max_train_time_minutes * 60
    
    print(f"Training will timeout after {max_train_time_minutes} minutes to prevent excessive runtime")
    
    # Data setup
    item_sequences = data['item_sequences']
    category_sequences = data['category_sequences']
    parent_category_sequences = data.get('parent_category_sequences', None)
    root_category_sequences = data.get('root_category_sequences', None)
    item_weights = data.get('item_weights', None)
    
    has_hierarchy = parent_category_sequences is not None and root_category_sequences is not None
    has_weights = item_weights is not None
    
    # Calculate model parameters - find max IDs safely
    max_item_id = 0
    for seq in item_sequences:
        for item in seq:
            max_item_id = max(max_item_id, item)
    
    max_category_id = 0
    for seq in category_sequences:
        for cat in seq:
            max_category_id = max(max_category_id, cat)
    
    num_items = max_item_id + 1
    num_categories = max_category_id + 1
    print(f"Vocabulary sizes: {num_items} items, {num_categories} categories")
    
    # Create model
    # Force CPU for better stability with large vocabulary sizes
    device = torch.device("cpu")
    model = SimpleRecommenderModel(
        num_items=num_items, 
        num_categories=num_categories,
        use_hierarchy=use_hierarchy and has_hierarchy,
        use_negative_sampling=use_negative_sampling
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    standard_loss_fn = nn.CrossEntropyLoss(reduction='none')  # We'll apply weights manually
    
    # Split sequences into train and validation sets
    train_size = int(len(item_sequences) * (1 - val_ratio))
    
    train_items = item_sequences[:train_size]
    train_categories = category_sequences[:train_size]
    
    val_items = item_sequences[train_size:]
    val_categories = category_sequences[train_size:]
    
    # Split hierarchical data if available
    train_parent_categories = parent_category_sequences[:train_size] if has_hierarchy else None
    train_root_categories = root_category_sequences[:train_size] if has_hierarchy else None
    
    val_parent_categories = parent_category_sequences[train_size:] if has_hierarchy else None
    val_root_categories = root_category_sequences[train_size:] if has_hierarchy else None
    
    # Split weights if available
    train_weights = item_weights[:train_size] if has_weights else None
    val_weights = item_weights[train_size:] if has_weights else None
    
    print(f"Split data: {len(train_items)} training sequences, {len(val_items)} validation sequences")
    
    # Create datasets and dataloaders
    train_dataset = SequenceDataset(
        train_items, train_categories, 
        train_parent_categories, train_root_categories, 
        train_weights, seq_length,
        neg_samples=neg_samples if use_negative_sampling else 0,
        item_vocab_size=num_items
    )
    
    val_dataset = SequenceDataset(
        val_items, val_categories, 
        val_parent_categories, val_root_categories,
        val_weights, seq_length,
        neg_samples=neg_samples if use_negative_sampling else 0,
        item_vocab_size=num_items
    )
    
    if len(train_dataset) == 0:
        print("Warning: No training samples generated. Check sequence length.")
        return model
        
    print(f"Created datasets with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_hit_rate = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            try:
                # Get batch data
                item_seq = batch['item_seq'].to(device)
                category_seq = batch['category_seq'].to(device)
                targets = batch['target'].to(device)
                seq_weights = batch['weight'].to(device)
                target_weights = batch['target_weight'].to(device)
                
                # Get hierarchical data if available
                parent_category_seq = batch.get('parent_category_seq', None)
                if parent_category_seq is not None:
                    parent_category_seq = parent_category_seq.to(device)
                
                root_category_seq = batch.get('root_category_seq', None)
                if root_category_seq is not None:
                    root_category_seq = root_category_seq.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                category_logits, item_logits = model(item_seq, category_seq, parent_category_seq, root_category_seq)
                
                # Check if target indices are within bounds
                valid_indices = targets < num_items
                if not torch.all(valid_indices):
                    print(f"Warning: Found {(~valid_indices).sum().item()} out-of-bounds target indices, skipping them")
                    # Filter valid indices
                    targets = targets[valid_indices]
                    item_logits = item_logits[valid_indices]
                    target_weights = target_weights[valid_indices]
                    
                    # Get negative samples if available and filter
                    if 'neg_samples' in batch and use_negative_sampling:
                        neg_samples = batch['neg_samples'][valid_indices].to(device)
                
                # Skip empty batches
                if len(targets) == 0:
                    continue
                
                # Handle loss calculation
                if use_negative_sampling and 'neg_samples' in batch:
                    # Get negative samples
                    neg_samples = batch['neg_samples'].to(device)
                    
                    # Compute loss with negative sampling
                    loss = model.compute_weighted_bce_loss(
                        item_logits, 
                        targets, 
                        neg_samples, 
                        target_weights
                    )
                else:
                    # Use standard cross entropy with weights
                    loss = standard_loss_fn(item_logits, targets)
                    
                    # Apply target weights (higher weights for cart/purchase events)
                    loss = (loss * target_weights).mean()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
        
        # Evaluation phase after each epoch
        if len(val_loader) > 0:
            model.eval()
            
            # Initialize metrics
            hit_count = 0
            mrr_sum = 0.0
            total_samples = 0
            weighted_hit_count = 0
            weighted_mrr_sum = 0.0
            weighted_total = 0.0
            
            for batch in val_loader:
                try:
                    # Get batch data
                    item_seq = batch['item_seq'].to(device)
                    category_seq = batch['category_seq'].to(device)
                    targets = batch['target'].to(device)
                    target_weights = batch['target_weight'].to(device)
                    
                    # Get hierarchical data if available
                    parent_category_seq = batch.get('parent_category_seq', None)
                    if parent_category_seq is not None:
                        parent_category_seq = parent_category_seq.to(device)
                    
                    root_category_seq = batch.get('root_category_seq', None)
                    if root_category_seq is not None:
                        root_category_seq = root_category_seq.to(device)
                    
                    # Forward pass
                    _, item_logits = model(item_seq, category_seq, parent_category_seq, root_category_seq)
                    
                    # Check if target indices are within bounds
                    valid_indices = targets < num_items
                    if not torch.all(valid_indices):
                        targets = targets[valid_indices]
                        item_logits = item_logits[valid_indices]
                        target_weights = target_weights[valid_indices]
                    
                    # Skip empty batches
                    if len(targets) == 0:
                        continue
                    
                    # Get top-10 predictions for each sample
                    _, top_items = torch.topk(item_logits, min(10, num_items-1), dim=1)
                    
                    # Calculate metrics for each sample in batch
                    for i in range(len(targets)):
                        target = targets[i].item()
                        weight = target_weights[i].item()
                        predictions = top_items[i].cpu().numpy().tolist()
                        
                        # Hit@10
                        hit = int(target in predictions)
                        hit_count += hit
                        weighted_hit_count += hit * weight
                        
                        # MRR
                        if hit:
                            rank = predictions.index(target) + 1
                            mrr = 1.0 / rank
                            mrr_sum += mrr
                            weighted_mrr_sum += mrr * weight
                        
                        total_samples += 1
                        weighted_total += weight
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
            
            # Calculate average metrics
            hit_rate = hit_count / total_samples if total_samples > 0 else 0
            mrr = mrr_sum / total_samples if total_samples > 0 else 0
            
            # Calculate weighted metrics
            weighted_hit_rate = weighted_hit_count / weighted_total if weighted_total > 0 else 0
            weighted_mrr = weighted_mrr_sum / weighted_total if weighted_total > 0 else 0

            print(f"  Validation: Hit@10 = {hit_rate:.4f}, MRR = {mrr:.4f} on {total_samples} samples")
            print(f"  Weighted Validation: Hit@10 = {weighted_hit_rate:.4f}, MRR = {weighted_mrr:.4f}")
            
            # Save best model based on weighted hit rate
            if weighted_hit_rate > best_hit_rate:
                best_hit_rate = weighted_hit_rate
                best_epoch = epoch + 1
        
        # Check for timeout
        if time.time() - start_time > max_train_time_seconds:
            print(f"Training timed out after {max_train_time_minutes} minutes")
            break
    
    print(f"\nBest model was from epoch {best_epoch} with weighted Hit@10 = {best_hit_rate:.4f}")
    
    return model

def load_data_from_file(data_path='data/', max_sessions=None, min_seq_length=2):
    """Load and process data from files in the data directory"""
    print(f"Loading data from {data_path}...")
    
    # Find available data files
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_path} not found")
    
    # Load all data files
    events_file = data_dir / 'events.csv'
    events = pd.read_csv(events_file)
    print(f"Loaded events data: {len(events)} rows")

    # Analyze event types
    event_types = events['event'].unique()
    print(f"Event types in dataset: {event_types}")
    
    # Define weights for different event types (higher weights = more important)
    event_type_weights = {
        'view': 1.0,               # Base weight for views
        'addtocart': 3.0,          # Add to cart shows stronger intent
        'transaction': 5.0,        # Transaction shows strongest intent
    }
    
    # Add weight column based on event type
    events['weight'] = events['event'].map(lambda x: event_type_weights.get(x, 1.0))
    print(f"Added weights to events based on event type")

    category_tree_file = data_dir / 'category_tree.csv'
    category_tree = pd.read_csv(category_tree_file)
    print(f"Loaded category tree: {len(category_tree)} categories")
    
    item_properties_file1 = data_dir / 'item_properties_part1.csv'
    item_properties_file2 = data_dir / 'item_properties_part2.csv'
    item_properties1 = pd.read_csv(item_properties_file1)
    item_properties2 = pd.read_csv(item_properties_file2)
    item_properties = pd.concat([item_properties1, item_properties2])
    print(f"Loaded item properties: {len(item_properties)} rows")
    
    # Process category hierarchy
    category_parents = {}
    category_levels = {}
    root_categories = set()
    
    # Extract category hierarchy relationships
    for _, row in category_tree.iterrows():
        category_id = row['categoryid']
        parent_id = row.get('parentid', None)
        category_parents[category_id] = parent_id
        
        # Track root categories (those without parents)
        if pd.isna(parent_id):
            root_categories.add(category_id)
            category_levels[category_id] = 0
    
    # Assign level to each category iteratively (faster than recursion)
    current_level = 0
    next_level_categories = list(root_categories)
    
    while next_level_categories:
        current_categories = next_level_categories
        next_level_categories = []
        current_level += 1
        
        for cat_id in current_categories:
            # Find all direct children of this category
            children = [c for c, p in category_parents.items() if p == cat_id and c not in category_levels]
            for child in children:
                category_levels[child] = current_level
                next_level_categories.append(child)
    
    print(f"Processed category hierarchy with {len(category_levels)} categories")
    
    # Extract item properties
    item_features = defaultdict(dict)
    
    # Extract category and other important properties
    for _, row in item_properties.iterrows():
        item_id = row['itemid']
        prop = row['property']
        value = row['value']
        
        # Store property values
        item_features[item_id][prop] = value
    
    # Extract item to category mapping from item_properties
    item_categories = {}
    item_parent_categories = {}  # Store parent categories for each item
    item_root_categories = {}    # Store root categories for each item
    
    category_props = item_properties[item_properties['property'].str.contains('category', case=False, na=False)]
    
    # Create item to category mapping with hierarchy
    for _, row in category_props.iterrows():
        try:
            item_id = row['itemid']
            category_id = int(row['value'])
            item_categories[item_id] = category_id
            
            # Find parent category
            parent_id = category_parents.get(category_id)
            if parent_id and not pd.isna(parent_id):
                item_parent_categories[item_id] = parent_id
            
            # Find root category by walking up the tree
            current = category_id
            while current in category_parents and category_parents[current] and not pd.isna(category_parents[current]):
                current = category_parents[current]
            item_root_categories[item_id] = current
            
        except (ValueError, TypeError):
            continue
    
    print(f"Extracted categories for {len(item_categories)} items")
    
    # Sort events by timestamp and group by visitor
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    events = events.sort_values(['visitorid', 'timestamp'])
    
    # Group by visitor to create sequences
    visitor_groups = events.groupby('visitorid')
    
    # Limit number of sessions if specified
    if max_sessions and max_sessions < len(visitor_groups):
        visitor_ids = list(visitor_groups.groups.keys())[:max_sessions]
        visitor_sessions = {vid: visitor_groups.get_group(vid) for vid in visitor_ids}
    else:
        visitor_sessions = {vid: group for vid, group in visitor_groups}
    
    # Create sequences with event type weighting and limited session augmentation
    item_sequences = []
    category_sequences = []
    parent_category_sequences = []  # Add parent category sequences
    root_category_sequences = []    # Add root category sequences
    item_weights = []  # Add weights for items based on event type
    skipped_sessions = 0
    augmented_sessions = 0
    
    # Limit augmentation parameters to reduce processing time
    session_max_length = 20  # Maximum length for sessions before splitting
    min_session_length = 3   # Minimum length for session augmentation
    max_augmentations_per_session = 2  # Limit number of augmentations per original session
    max_total_augmentations = min(500, len(visitor_sessions) // 2)  # Cap total augmentations
    
    # Track how many augmentations we've created
    total_augmentations = 0
    
    for visitor_id, session in visitor_sessions.items():
        # Extract item IDs and weights in sequence
        items = session['itemid'].tolist()
        weights = session['weight'].tolist()
        
        # Skip very short sequences that can't form at least one input-target pair
        if len(items) < min_seq_length:
            skipped_sessions += 1
            continue
        
        # Get category for each item
        categories = []
        parent_categories = []
        root_categories = []
        
        for item in items:
            # Use category if available, otherwise use default (1)
            category = item_categories.get(item, 1)
            categories.append(category)
            
            # Get parent category, use default (1) if not available
            parent_category = item_parent_categories.get(item, 1)
            parent_categories.append(parent_category)
            
            # Get root category, use default (1) if not available
            root_category = item_root_categories.get(item, 1)
            root_categories.append(root_category)
        
        # Add original sequence
        item_sequences.append(items)
        category_sequences.append(categories)
        parent_category_sequences.append(parent_categories)
        root_category_sequences.append(root_categories)
        item_weights.append(weights)
        
        # Stop augmentation if we've reached the limit
        if total_augmentations >= max_total_augmentations:
            continue
            
        # Track augmentations for this session
        session_augmentations = 0
        
        # Session augmentation: split long sessions into sub-sessions
        if len(items) > session_max_length and session_augmentations < max_augmentations_per_session:
            # Just take beginning and end of very long sessions
            # Beginning
            begin_items = items[:session_max_length]
            begin_categories = categories[:session_max_length]
            begin_parent_categories = parent_categories[:session_max_length]
            begin_root_categories = root_categories[:session_max_length]
            begin_weights = weights[:session_max_length]
            
            item_sequences.append(begin_items)
            category_sequences.append(begin_categories)
            parent_category_sequences.append(begin_parent_categories)
            root_category_sequences.append(begin_root_categories)
            item_weights.append(begin_weights)
            
            # End
            end_items = items[-session_max_length:]
            end_categories = categories[-session_max_length:]
            end_parent_categories = parent_categories[-session_max_length:]
            end_root_categories = root_categories[-session_max_length:]
            end_weights = weights[-session_max_length:]
            
            item_sequences.append(end_items)
            category_sequences.append(end_categories)
            parent_category_sequences.append(end_parent_categories)
            root_category_sequences.append(end_root_categories)
            item_weights.append(end_weights)
            
            session_augmentations += 2
            total_augmentations += 2
            augmented_sessions += 2
        
        # Session augmentation: create synthetic sessions with high-weight items (purchases, cart adds)
        if len(items) >= min_session_length and total_augmentations < max_total_augmentations and session_augmentations < max_augmentations_per_session:
            # Find high-weight items (add to cart, purchase)
            high_weight_indices = [i for i, w in enumerate(weights) if w > 1.0]
            
            if len(high_weight_indices) >= 2:
                # Create a new session focusing on high-weight items
                high_items = [items[i] for i in high_weight_indices]
                high_categories = [categories[i] for i in high_weight_indices]
                high_parent_categories = [parent_categories[i] for i in high_weight_indices]
                high_root_categories = [root_categories[i] for i in high_weight_indices]
                high_weights = [weights[i] for i in high_weight_indices]
                
                item_sequences.append(high_items)
                category_sequences.append(high_categories)
                parent_category_sequences.append(high_parent_categories)
                root_category_sequences.append(high_root_categories)
                item_weights.append(high_weights)
                
                session_augmentations += 1
                total_augmentations += 1
                augmented_sessions += 1
    
    print(f"Created {len(item_sequences)} visitor sequences ({augmented_sessions} from augmentation), skipped {skipped_sessions} sessions with fewer than {min_seq_length} items")
    
    return {
        'item_sequences': item_sequences,
        'category_sequences': category_sequences,
        'parent_category_sequences': parent_category_sequences,
        'root_category_sequences': root_category_sequences,
        'item_weights': item_weights,
        'item_categories': item_categories,
        'category_hierarchy': category_parents,
        'category_levels': category_levels
    }


def main():
    """Main function to demonstrate model usage"""
    try:
        # Load data with lightweight settings to speed up processing
        print("Loading data with optimized settings for faster performance...")
        data = load_data_from_file(data_path='data/', max_sessions=500, min_seq_length=2)
        
        # Train model with enhanced features but optimized for speed
        print("\nTraining model with optimized settings...")
        model = train_simple_model(
            data, 
            epochs=3,                    # Reduced epochs for faster training
            seq_length=2,                # Shorter sequence length
            batch_size=32,               # Standard batch size
            learning_rate=0.001,         # Standard learning rate
            use_hierarchy=True,          # Keep hierarchy features
            use_negative_sampling=True,  # Keep negative sampling
            neg_samples=3,               # Reduced negative samples for speed
            max_train_time_minutes=15    # Add timeout to prevent excessive runtime
        )

        print("\nModel training complete with optimized parameters!")
        print("The model uses:")
        print("1. Category hierarchy - optimized with iterative processing")
        print("2. Event type weighting - prioritizing purchases and cart additions")
        print("3. Session augmentation - with limits to improve performance")
        print("4. Negative sampling - with reduced samples for better speed")
        print("5. Runtime protection - training timeout after 15 minutes")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")


if __name__ == "__main__":
    main()