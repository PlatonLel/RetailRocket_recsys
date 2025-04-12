import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import math
from typing import Dict, List, Tuple, Optional, Union
import os
import random


class ECommerceRecommenderModel(nn.Module):
    """
    A comprehensive e-commerce recommendation model with multiple components.
    
    This model integrates multiple prediction approaches including Markov processes,
    hierarchical predictions, LSTM-based session modeling, and embeddings to provide
    accurate recommendations for e-commerce applications.
    
    Components:
    - Markov Process: For predicting state transitions
    - Two-Level Hierarchy: Managing category and item level predictions
    - LSTM Layer: Capturing session context over time
    - Embedding Layers: Encoding contexts meaningfully
    - Category Predictor: Forecasting next item category
    - Item Predictor: Forecasting specific item within a category
    - Next User Move Predictor: Anticipating user's next action
    """
    
    def __init__(
        self, 
        num_items: int,
        num_categories: int,
        num_actions: int, 
        embedding_dim: int = 128,
        lstm_hidden_dim: int = 128,
        markov_order: int = 1,
        dropout_rate: float = 0.2,
        max_items_for_markov: int = 10000,
        use_candidate_sampling: bool = True,
        candidate_sample_size: int = 1000,
        items_by_category: Optional[Dict[int, List[int]]] = None
    ):
        """
        Initialize the E-Commerce Recommender Model.
        
        Args:
            num_items: Total number of unique items
            num_categories: Total number of unique categories
            num_actions: Number of possible user actions (e.g., view, add to cart, purchase)
            embedding_dim: Dimension of embedding vectors
            lstm_hidden_dim: Hidden dimension of LSTM layers
            markov_order: Order of the Markov process
            dropout_rate: Dropout rate for regularization
            max_items_for_markov: Maximum number of items to consider for Markov process
            use_candidate_sampling: Whether to use candidate sampling for item prediction
            candidate_sample_size: Number of candidates to sample for item prediction
            items_by_category: Optional dictionary mapping category IDs to lists of item IDs
        """
        super(ECommerceRecommenderModel, self).__init__()
        
        self.num_items = num_items
        self.num_categories = num_categories
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.markov_order = markov_order
        self.max_items_for_markov = max_items_for_markov
        self.use_candidate_sampling = use_candidate_sampling
        self.candidate_sample_size = candidate_sample_size
        self.items_by_category = items_by_category or {}
        
        # Track co-occurrence information for candidate sampling
        self.item_co_occurrence = defaultdict(Counter)
        
        self.item_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.category_embeddings = nn.Embedding(num_categories + 1, embedding_dim, padding_idx=0)
        self.action_embeddings = nn.Embedding(num_actions + 1, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 3, 
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )
        
        hidden_layer_size = lstm_hidden_dim * 2
        
        self.category_predictor_hidden = nn.Linear(lstm_hidden_dim, hidden_layer_size)
        self.category_predictor_output = nn.Linear(hidden_layer_size, num_categories)
        
        self.item_predictor_hidden = nn.Linear(lstm_hidden_dim + embedding_dim, hidden_layer_size)
        self.item_predictor_output = nn.Linear(hidden_layer_size, num_items)
        
        self.next_move_predictor_hidden = nn.Linear(lstm_hidden_dim, hidden_layer_size)
        self.next_move_predictor_output = nn.Linear(hidden_layer_size, num_actions)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        # Markov process transition matrices
        self._initialize_markov_matrices()
        
    def _initialize_markov_matrices(self):
        """
        Initialize the Markov process transition matrices using dictionaries for efficiency.
        """
        # Use dictionaries for sparse representation
        self.item_transitions = defaultdict(Counter)
        self.category_transitions = defaultdict(Counter)
        self.action_transitions = defaultdict(Counter)
        
        # For higher-order Markov
        self.higher_order_transitions = {
            'item': defaultdict(Counter),
            'category': defaultdict(Counter),
            'action': defaultdict(Counter)
        }
        
        # Keep track of popular items for Markov predictions
        self.popular_items = set()
        self.item_frequency = Counter()
        
    def update_markov_transitions(self, sequences: List[List[int]], entity_type: str = 'item'):
        """
        Update Markov transition probabilities based on observed sequences.
        
        Args:
            sequences: List of sequences (e.g., item sequences, category sequences)
            entity_type: Type of entity ('item', 'category', or 'action')
        """
        if entity_type == 'item':
            transitions = self.item_transitions
            for sequence in sequences:
                for item in sequence:
                    self.item_frequency[item] += 1
            self.popular_items = set([item for item, _ in self.item_frequency.most_common(self.max_items_for_markov)])
        elif entity_type == 'category':
            transitions = self.category_transitions
        elif entity_type == 'action':
            transitions = self.action_transitions
        else:
            raise ValueError("entity_type must be 'item', 'category', or 'action'")
        
        for sequence in sequences:
            if entity_type == 'item':
                filtered_sequence = [item for item in sequence if item in self.popular_items or len(self.popular_items) < self.max_items_for_markov]
                if len(filtered_sequence) < 2:
                    continue
                sequence = filtered_sequence
                
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_entity = sequence[i+1]
                transitions[current][next_entity] += 1
        
        if self.markov_order > 1:
            self._update_higher_order_transitions(sequences, entity_type)
            
    def _update_higher_order_transitions(self, sequences: List[List[int]], entity_type: str):
        """
        Update higher-order Markov transition probabilities.
        
        Args:
            sequences: List of sequences
            entity_type: Type of entity ('item', 'category', or 'action')
        """
        transition_counts = self.higher_order_transitions[entity_type]
        
        if entity_type == 'item' and len(self.popular_items) > 0:
            filtered_sequences = []
            for sequence in sequences:
                filtered_seq = [item for item in sequence if item in self.popular_items or len(self.popular_items) < self.max_items_for_markov]
                if len(filtered_seq) >= self.markov_order + 1:
                    filtered_sequences.append(filtered_seq)
            sequences = filtered_sequences
        
        for sequence in sequences:
            for i in range(len(sequence) - self.markov_order):
                context = tuple(sequence[i:i+self.markov_order])
                next_entity = sequence[i+self.markov_order]
                transition_counts[context][next_entity] += 1
    
    def predict_next_with_markov(self, sequence: List[int], entity_type: str = 'item') -> int:
        """
        Predict the next entity (item, category, action) using Markov process.
        
        Args:
            sequence: Recent sequence of entities
            entity_type: Type of entity ('item', 'category', or 'action')
            
        Returns:
            Predicted next entity id
        """
        if entity_type == 'item':
            sequence = [item for item in sequence if item in self.popular_items or len(self.popular_items) < self.max_items_for_markov]
            if not sequence:
                if self.item_frequency:
                    return self.item_frequency.most_common(1)[0][0]
                return 0
        
        if self.markov_order > 1 and len(sequence) >= self.markov_order:
            return self._predict_higher_order_markov(sequence, entity_type)
        
        if not sequence:
            return 0 
        
        current = sequence[-1]
        
        if entity_type == 'item':
            transitions = self.item_transitions
        elif entity_type == 'category':
            transitions = self.category_transitions
        elif entity_type == 'action':
            transitions = self.action_transitions
        else:
            raise ValueError("entity_type must be 'item', 'category', or 'action'")
        
        next_entities = transitions.get(current, Counter())
        
        if next_entities:
            return max(next_entities.items(), key=lambda x: x[1])[0]
        
        return 0
    
    def _predict_higher_order_markov(self, sequence: List[int], entity_type: str) -> int:
        """
        Make a prediction using higher-order Markov model.
        
        Args:
            sequence: Recent sequence of entities
            entity_type: Type of entity
            
        Returns:
            Predicted next entity id
        """
        context = tuple(sequence[-self.markov_order:])
        transitions = self.higher_order_transitions[entity_type]
        
        if context in transitions and transitions[context]:
            return max(transitions[context].items(), key=lambda x: x[1])[0]
        
        return self.predict_next_with_markov([sequence[-1]], entity_type)
    
    def update_item_co_occurrence(self, item_sequences: List[List[int]], window_size: int = 5):
        """
        Update item co-occurrence counts for candidate sampling.
        
        Args:
            item_sequences: List of item ID sequences
            window_size: Size of co-occurrence window
        """
        for sequence in item_sequences:
            for i in range(len(sequence)):
                current_item = sequence[i]
                window_start = max(0, i - window_size)
                window_end = min(len(sequence), i + window_size + 1)
                
                for j in range(window_start, window_end):
                    if i != j: 
                        co_occurring_item = sequence[j]
                        self.item_co_occurrence[current_item][co_occurring_item] += 1
    
    def update_items_by_category(self, category_sequences: List[List[int]], item_sequences: List[List[int]]):
        """
        Update mapping of categories to items.
        
        Args:
            category_sequences: List of category ID sequences
            item_sequences: List of item ID sequences
        """
        for i in range(len(category_sequences)):
            category_seq = category_sequences[i]
            item_seq = item_sequences[i]
            
            for j in range(len(category_seq)):
                category_id = category_seq[j]
                item_id = item_seq[j]
                
                if category_id not in self.items_by_category:
                    self.items_by_category[category_id] = []
                
                if item_id not in self.items_by_category[category_id]:
                    self.items_by_category[category_id].append(item_id)
    
    def get_item_candidates(self, item_sequence: torch.Tensor, category_id: Optional[int] = None) -> List[int]:
        """
        Get candidate items for prediction based on co-occurrence and category.
        
        Args:
            item_sequence: Sequence of item IDs
            category_id: Optional category ID to filter candidates
            
        Returns:
            List of candidate item IDs
        """
        candidates = set()
        
        if torch.is_tensor(item_sequence):
            item_sequence = item_sequence.cpu().numpy().tolist()
        
        if category_id is not None and category_id in self.items_by_category:
            category_items = self.items_by_category[category_id]
            candidates.update(category_items)
        
        for item_id in item_sequence:
            if item_id in self.item_co_occurrence:
                co_occurring_items = [item for item, _ in self.item_co_occurrence[item_id].most_common(100)]
                candidates.update(co_occurring_items)
        
        if len(candidates) < self.candidate_sample_size:
            popular_items = [item for item, _ in self.item_frequency.most_common(self.candidate_sample_size)]
            candidates.update(popular_items)
        
        if len(candidates) < self.candidate_sample_size:
            random_items = random.sample(
                range(self.num_items), 
                min(self.candidate_sample_size - len(candidates), self.num_items)
            )
            candidates.update(random_items)
        
        candidates_list = list(candidates)[:self.candidate_sample_size]
        candidates_list = [c for c in candidates_list if c < self.num_items]
        
        if len(candidates_list) < 1:
            candidates_list = [0] 
        
        return candidates_list
    
    def forward_with_candidates(
        self, 
        item_sequences: torch.Tensor,
        category_sequences: torch.Tensor,
        action_sequences: torch.Tensor,
        candidates: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with candidate sampling for efficient item prediction.
        
        Args:
            item_sequences: Batch of item id sequences [batch_size, seq_len]
            category_sequences: Batch of category id sequences [batch_size, seq_len]
            action_sequences: Batch of action id sequences [batch_size, seq_len]
            candidates: Optional list of candidate item IDs
            
        Returns:
            tuple of (category_logits, item_logits, action_logits)
        """
        category_logits, _, action_logits = self.forward(
            item_sequences, category_sequences, action_sequences
        )
        
        if candidates is None:
            batch_candidates = []
            for i in range(item_sequences.size(0)):
                seq_candidates = self.get_item_candidates(
                    item_sequences[i],
                    category_sequences[i, -1].item()
                )
                batch_candidates.append(seq_candidates)
        else:
            batch_candidates = [candidates] * item_sequences.size(0)
        
        item_embeds = self.item_embeddings(item_sequences)
        category_embeds = self.category_embeddings(category_sequences)
        action_embeds = self.action_embeddings(action_sequences)
        
        combined_embeds = torch.cat(
            [item_embeds, category_embeds, action_embeds], 
            dim=2
        )
        
        lstm_out, _ = self.lstm(combined_embeds)
        lstm_final = lstm_out[:, -1, :] 
        lstm_final = self.dropout(lstm_final)
        
        category_probs = torch.softmax(category_logits, dim=1)
        predicted_category_embed = torch.matmul(
            category_probs, 
            self.category_embeddings.weight[1:self.num_categories+1]
        )
        
        item_input = torch.cat([lstm_final, predicted_category_embed], dim=1)
        item_hidden = self.relu(self.item_predictor_hidden(item_input))
        item_hidden = self.dropout(item_hidden)
        
        batch_item_logits = []
        for i in range(item_input.size(0)):
            valid_candidates = [c for c in batch_candidates[i] if c < self.num_items]
            if not valid_candidates:
                valid_candidates = [0] 
                
            candidate_ids = torch.tensor(valid_candidates, device=item_input.device)
            candidate_weights = self.item_predictor_output.weight[candidate_ids]
            candidate_biases = self.item_predictor_output.bias[candidate_ids]
            
            candidate_logits = torch.matmul(item_hidden[i:i+1], candidate_weights.t()) + candidate_biases
            
            full_logits = torch.full((1, self.num_items), -1e10, device=item_input.device)
            full_logits[0, candidate_ids] = candidate_logits
            
            batch_item_logits.append(full_logits)
        
        item_logits = torch.cat(batch_item_logits, dim=0)
        
        return category_logits, item_logits, action_logits
    
    def forward(
        self, 
        item_sequences: torch.Tensor,
        category_sequences: torch.Tensor,
        action_sequences: torch.Tensor,
        predict_category: bool = True,
        category_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            item_sequences: Batch of item id sequences [batch_size, seq_len]
            category_sequences: Batch of category id sequences [batch_size, seq_len]
            action_sequences: Batch of action id sequences [batch_size, seq_len] 
            predict_category: Whether to predict category (True) or use provided category (False)
            category_input: Optional tensor of category ids for item prediction
            
        Returns:
            tuple of (category_logits, item_logits, action_logits)
        """
        batch_size, seq_len = item_sequences.shape
        
        item_embeds = self.item_embeddings(item_sequences)
        category_embeds = self.category_embeddings(category_sequences)
        action_embeds = self.action_embeddings(action_sequences)
        
        combined_embeds = torch.cat(
            [item_embeds, category_embeds, action_embeds], 
            dim=2
        )
        
        lstm_out, (hidden, cell) = self.lstm(combined_embeds)
        lstm_final = lstm_out[:, -1, :] 
        lstm_final = self.dropout(lstm_final)
        
        category_hidden = self.relu(self.category_predictor_hidden(lstm_final))
        category_hidden = self.dropout(category_hidden)
        category_logits = self.category_predictor_output(category_hidden)
        
        if predict_category:
            category_probs = torch.softmax(category_logits, dim=1)
            predicted_category_embed = torch.matmul(
                category_probs, 
                self.category_embeddings.weight[1:self.num_categories+1] 
            )
        else:
            predicted_category_embed = self.category_embeddings(category_input[:, -1])
        
        item_input = torch.cat([lstm_final, predicted_category_embed], dim=1)
        item_hidden = self.relu(self.item_predictor_hidden(item_input))
        item_hidden = self.dropout(item_hidden)
        item_logits = self.item_predictor_output(item_hidden)
        
        next_move_hidden = self.relu(self.next_move_predictor_hidden(lstm_final))
        next_move_hidden = self.dropout(next_move_hidden)
        next_move_logits = self.next_move_predictor_output(next_move_hidden)
        
        return category_logits, item_logits, next_move_logits
    
    def get_category_prediction(
        self, 
        item_sequences: torch.Tensor,
        category_sequences: torch.Tensor,
        action_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Get category predictions only.
        
        Args:
            item_sequences: Batch of item id sequences
            category_sequences: Batch of category id sequences
            action_sequences: Batch of action id sequences
            
        Returns:
            Category prediction logits
        """
        category_logits, _, _ = self.forward(
            item_sequences, category_sequences, action_sequences
        )
        return category_logits
    
    def get_item_prediction(
        self, 
        item_sequences: torch.Tensor,
        category_sequences: torch.Tensor,
        action_sequences: torch.Tensor,
        category_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get item predictions only.
        
        Args:
            item_sequences: Batch of item id sequences
            category_sequences: Batch of category id sequences
            action_sequences: Batch of action id sequences
            category_input: Optional tensor of category ids for item prediction
            
        Returns:
            Item prediction logits
        """
        _, item_logits, _ = self.forward(
            item_sequences, 
            category_sequences, 
            action_sequences,
            predict_category=category_input is None,
            category_input=category_input
        )
        return item_logits
    
    def get_next_action_prediction(
        self, 
        item_sequences: torch.Tensor,
        category_sequences: torch.Tensor,
        action_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Get next user action predictions only.
        
        Args:
            item_sequences: Batch of item id sequences
            category_sequences: Batch of category id sequences
            action_sequences: Batch of action id sequences
            
        Returns:
            Next action prediction logits
        """
        _, _, action_logits = self.forward(
            item_sequences, category_sequences, action_sequences
        )
        return action_logits


class ECommerceRecommenderTrainer:
    """
    Trainer class for the ECommerceRecommenderModel.
    
    This class handles data processing, model training, and evaluation.
    """
    
    def __init__(
        self,
        model: ECommerceRecommenderModel,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_weighted_loss: bool = True
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The ECommerceRecommenderModel to train
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            device: Device to train on ('cuda' or 'cpu')
            use_weighted_loss: Whether to use weighted loss functions for imbalanced data
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.use_weighted_loss = use_weighted_loss
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )
        
        self.category_loss_fn = nn.CrossEntropyLoss()
        self.item_loss_fn = nn.CrossEntropyLoss()
        self.action_loss_fn = nn.CrossEntropyLoss()
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        self.patience = 3
    
    def prepare_data(self, item_sequences, category_sequences, action_sequences, test_size=0.2, eval_size=0.1, random_state=42):
        """
        Prepare data splits for training, testing, and evaluation.
        
        Args:
            item_sequences: List of item ID sequences
            category_sequences: List of category ID sequences
            action_sequences: List of action ID sequences
            test_size: Proportion of data to use for testing
            eval_size: Proportion of data to use for evaluation
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary of data splits
        """
        total_sequences = len(item_sequences)
        adjusted_eval_size = eval_size / (1 - test_size) 
        
        train_items, test_items, train_categories, test_categories, train_actions, test_actions = train_test_split(
            item_sequences, category_sequences, action_sequences,
            test_size=test_size, random_state=random_state
        )
        
        train_items, eval_items, train_categories, eval_categories, train_actions, eval_actions = train_test_split(
            train_items, train_categories, train_actions,
            test_size=adjusted_eval_size, random_state=random_state
        )
        
        data_splits = {
            'train': {
                'items': train_items,
                'categories': train_categories,
                'actions': train_actions
            },
            'test': {
                'items': test_items,
                'categories': test_categories,
                'actions': test_actions
            },
            'eval': {
                'items': eval_items,
                'categories': eval_categories,
                'actions': eval_actions
            }
        }
        
        return data_splits
    
    def train_epoch(self, train_loader, epoch, log_interval=100):
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader with training data
            epoch: Current epoch number
            log_interval: How often to log progress
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            item_sequences = batch['item_input'].to(self.device)
            category_sequences = batch['category_input'].to(self.device)
            action_sequences = batch['action_input'].to(self.device)
            
            item_targets = batch['item_target'].to(self.device)
            category_targets = batch['category_target'].to(self.device)
            action_targets = batch['action_target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.model.use_candidate_sampling:
                category_logits, item_logits, action_logits = self.model.forward_with_candidates(
                    item_sequences, category_sequences, action_sequences
                )
            else:
                category_logits, item_logits, action_logits = self.model(
                    item_sequences, category_sequences, action_sequences
                )
            
            category_loss = self.category_loss_fn(category_logits, category_targets)
            item_loss = self.item_loss_fn(item_logits, item_targets)
            action_loss = self.action_loss_fn(action_logits, action_targets)
            
            loss = category_loss + item_loss + action_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(item_sequences)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]	Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / batch_count
        
        return {'loss': avg_loss}
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader with evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in data_loader:
                item_sequences = batch['item_input'].to(self.device)
                category_sequences = batch['category_input'].to(self.device)
                action_sequences = batch['action_input'].to(self.device)
                
                item_targets = batch['item_target'].to(self.device)
                category_targets = batch['category_target'].to(self.device)
                action_targets = batch['action_target'].to(self.device)
                
                if self.model.use_candidate_sampling:
                    category_logits, item_logits, action_logits = self.model.forward_with_candidates(
                        item_sequences, category_sequences, action_sequences
                    )
                else:
                    category_logits, item_logits, action_logits = self.model(
                        item_sequences, category_sequences, action_sequences
                    )
                
                category_loss = self.category_loss_fn(category_logits, category_targets)
                item_loss = self.item_loss_fn(item_logits, item_targets)
                action_loss = self.action_loss_fn(action_logits, action_targets)
                
                loss = category_loss + item_loss + action_loss
                
                total_loss += loss.item()
                batch_count += 1
        
        avg_loss = total_loss / batch_count
        
        return {'loss': avg_loss}
    
    def compute_class_weights(
        self, 
        targets: List[int], 
        num_classes: int, 
        smoothing_factor: float = 0.1
    ) -> torch.Tensor:
        """
        Compute class weights based on class frequencies.
        
        Args:
            targets: List of target class indices
            num_classes: Total number of classes
            smoothing_factor: Smoothing factor to avoid extreme weights
            
        Returns:
            Tensor of class weights
        """
        class_counts = Counter(targets)
        
        for i in range(num_classes):
            if i not in class_counts:
                class_counts[i] = 0
        
        total_samples = len(targets)
        weights = []
        
        for i in range(num_classes):
            count = class_counts[i]
            if count == 0:
                weights.append(1.0) 
            else:
                weight = (total_samples / (count + smoothing_factor * total_samples))
                weights.append(weight)
        
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        weights_tensor = weights_tensor / weights_tensor.sum() * num_classes
        
        return weights_tensor
    
    def setup_weighted_loss_functions(
        self, 
        category_targets: List[int], 
        item_targets: List[int], 
        action_targets: List[int]
    ):
        """
        Set up weighted loss functions based on class frequencies.
        
        Args:
            category_targets: List of category target indices
            item_targets: List of item target indices
            action_targets: List of action target indices
        """
        if not self.use_weighted_loss:
            return
        
        category_weights = self.compute_class_weights(
            category_targets, 
            self.model.num_categories
        )
        
        # Item weights are generally not computed due to sparsity; candidate sampling is used.
        
        action_weights = self.compute_class_weights(
            action_targets, 
            self.model.num_actions
        )
        
        self.category_loss_fn = nn.CrossEntropyLoss(
            weight=category_weights.to(self.device)
        )
        
        self.action_loss_fn = nn.CrossEntropyLoss(
            weight=action_weights.to(self.device)
        )
        
        print("Using weighted loss functions.")
        # Optionally print weights if needed for debugging, but can be verbose
        # print(f"  Category weights: {category_weights}")
        # print(f"  Action weights: {action_weights}")
    
    def calculate_recall_at_k(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        k: int = 10
    ) -> float:
        """
        Calculate Recall@K metric.
        
        Args:
            predictions: Prediction logits [batch_size, num_items]
            targets: Target item indices [batch_size]
            k: K value for Recall@K
            
        Returns:
            Recall@K score
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        targets_expanded = targets.unsqueeze(1).expand_as(top_k_indices)
        hits = (top_k_indices == targets_expanded).float()
        recall = hits.sum(dim=1) / 1.0 # Recall is 1 if hit, 0 otherwise for single target
        return recall.mean().item()

    
    def calculate_mrr(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        k: int = 10
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) metric.
        
        Args:
            predictions: Prediction logits [batch_size, num_items]
            targets: Target item indices [batch_size]
            k: Maximum rank to consider
            
        Returns:
            MRR score
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        targets_expanded = targets.unsqueeze(1).expand_as(top_k_indices)
        hits = (top_k_indices == targets_expanded).float()
        
        hit_indices = torch.nonzero(hits, as_tuple=True)
        ranks = hit_indices[1] + 1 
        reciprocal_ranks = 1.0 / ranks.float()
        
        batch_size = predictions.size(0)
        all_reciprocal_ranks = torch.zeros(batch_size, device=predictions.device)
        
        # Ensure we only assign reciprocal ranks for batches that had a hit
        batch_indices_with_hits = hit_indices[0]
        if batch_indices_with_hits.numel() > 0:
             # Need to handle cases where a target might appear multiple times in top-k (take first occurrence)
             unique_batch_indices, first_occurrence_indices = torch.unique(batch_indices_with_hits, return_inverse=True)
             # This logic is complex if multiple hits per batch; simplifying for typical use case
             # Assuming one target per row, find the first hit index
             first_hit_mask = torch.cat((torch.tensor([True], device=hits.device), hits.sum(dim=1)[:-1])) > 0
             first_hit_batch_indices = batch_indices_with_hits[first_occurrence_indices]

             # Correct assignment needed
             unique_hits, hit_batch_idx = torch.unique(hit_indices[0], return_inverse=False, return_counts=False)
             # Find the ranks corresponding to the first hit for each batch item
             first_hit_ranks = torch.zeros_like(unique_hits, dtype=torch.float)
             for i, batch_idx in enumerate(unique_hits):
                  batch_hits_indices = (hit_indices[0] == batch_idx).nonzero(as_tuple=True)[0]
                  first_hit_rank = ranks[batch_hits_indices[0]]
                  first_hit_ranks[i] = 1.0 / first_hit_rank.float()

             all_reciprocal_ranks[unique_hits] = first_hit_ranks

        mrr = all_reciprocal_ranks.mean().item()
        return mrr

    
    def calculate_ndcg(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) metric.
        
        Args:
            predictions: Prediction logits [batch_size, num_items]
            targets: Target item indices [batch_size]
            k: K value for NDCG@K
            
        Returns:
            NDCG@K score
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        targets_expanded = targets.unsqueeze(1).expand_as(top_k_indices)
        relevance = (top_k_indices == targets_expanded).float()
        
        position = torch.arange(1, k + 1, dtype=torch.float, device=predictions.device)
        discount = torch.log2(position + 1)
        dcg = (relevance / discount.unsqueeze(0)).sum(dim=1)
        
        # IDCG is 1/log2(1+1) = 1 if the target is present in top-k, 0 otherwise
        # Simplified: IDCG is 1 because there's only one relevant item (the target).
        # If the target is in the top-k, the ideal DCG is 1/log2(1+1)=1. Otherwise, NDCG is 0.
        # A more standard way considers relevance scores if they were available.
        # Here, relevance is binary.
        idcg = torch.ones_like(dcg) # Ideal DCG is 1 if target exists
        
        # Ensure IDCG is 0 if the target is not in the top k predictions at all
        target_in_top_k = relevance.sum(dim=1) > 0
        idcg[~target_in_top_k] = 1.0 # Avoid division by zero; if DCG is 0, NDCG will be 0.

        ndcg = dcg / idcg
        
        avg_ndcg = ndcg.mean().item()
        return avg_ndcg

    
    def evaluate_advanced_metrics(
        self,
        data_loader: torch.utils.data.DataLoader,
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, float]:
        """
        Evaluate model with advanced ranking metrics.
        
        Args:
            data_loader: DataLoader with evaluation data
            k_values: List of K values for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        metrics = {}
        
        for k in k_values:
            metrics[f'recall@{k}'] = 0.0
            metrics[f'mrr@{k}'] = 0.0
            metrics[f'ndcg@{k}'] = 0.0
        
        metrics['category_accuracy'] = 0.0
        metrics['item_accuracy'] = 0.0
        metrics['action_accuracy'] = 0.0
        
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                item_sequences = batch['item_input'].to(self.device)
                category_sequences = batch['category_input'].to(self.device)
                action_sequences = batch['action_input'].to(self.device)
                
                item_targets = batch['item_target'].to(self.device)
                category_targets = batch['category_target'].to(self.device)
                action_targets = batch['action_target'].to(self.device)
                
                if self.model.use_candidate_sampling:
                    category_logits, item_logits, action_logits = self.model.forward_with_candidates(
                        item_sequences, category_sequences, action_sequences
                    )
                else:
                    category_logits, item_logits, action_logits = self.model(
                        item_sequences, category_sequences, action_sequences
                    )
                
                category_pred = torch.argmax(category_logits, dim=1)
                item_pred = torch.argmax(item_logits, dim=1)
                action_pred = torch.argmax(action_logits, dim=1)
                
                batch_size = item_targets.size(0)
                total_samples += batch_size
                
                metrics['category_accuracy'] += (category_pred == category_targets).sum().item()
                metrics['item_accuracy'] += (item_pred == item_targets).sum().item() # Top-1 accuracy
                metrics['action_accuracy'] += (action_pred == action_targets).sum().item()
                
                for k in k_values:
                    if k <= item_logits.size(1):
                        # Note: Recall@k calculation needed adjustment
                        _, top_k_preds = torch.topk(item_logits, k, dim=1)
                        hits_k = (top_k_preds == item_targets.unsqueeze(1)).sum(dim=1) > 0
                        metrics[f'recall@{k}'] += hits_k.float().sum().item()
                        
                        # MRR and NDCG calculations need careful implementation
                        # Using the previously defined functions, but ensuring correctness
                        metrics[f'mrr@{k}'] += self.calculate_mrr(item_logits, item_targets, k) * batch_size
                        metrics[f'ndcg@{k}'] += self.calculate_ndcg(item_logits, item_targets, k) * batch_size
        
        for key in metrics:
            metrics[key] /= total_samples
        
        return metrics

    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        eval_loader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        log_interval: int = 100,
        early_stopping: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader with training data
            test_loader: DataLoader with test data
            eval_loader: DataLoader with evaluation data
            num_epochs: Number of epochs to train for
            log_interval: How often to log progress
            early_stopping: Whether to use early stopping
            
        Returns:
            Dictionary of training and evaluation metrics
        """
        if self.use_weighted_loss:
            print("Setting up weighted loss functions...")
            all_category_targets = []
            all_item_targets = [] # Not used for weights usually
            all_action_targets = []
            
            for batch in train_loader:
                all_category_targets.extend(batch['category_target'].numpy().tolist())
                # all_item_targets.extend(batch['item_target'].numpy().tolist()) # Item weights not typically used
                all_action_targets.extend(batch['action_target'].numpy().tolist())
            
            self.setup_weighted_loss_functions(
                all_category_targets,
                [], # Pass empty list for items if not used
                all_action_targets
            )
        
        metrics = defaultdict(list)
        
        for epoch in range(1, num_epochs + 1):
            print(f'Epoch {epoch}/{num_epochs}')
            
            train_metrics = self.train_epoch(train_loader, epoch, log_interval)
            metrics['train_loss'].append(train_metrics['loss'])
            
            test_metrics = self.evaluate(test_loader)
            metrics['test_loss'].append(test_metrics['loss'])
            
            eval_metrics = self.evaluate_advanced_metrics(eval_loader)
            for key, value in eval_metrics.items():
                 metrics[key].append(value)
            metrics['eval_loss'].append(test_metrics['loss']) # Use test loss for eval loss tracking
            
            self.scheduler.step(test_metrics['loss'])
            
            print(f'Train Loss: {train_metrics["loss"]:.4f}')
            print(f'Test Loss: {test_metrics["loss"]:.4f}')
            print(f'Eval Metrics: ' + ', '.join([f'{k}: {v:.4f}' for k, v in eval_metrics.items()]))

            
            if early_stopping:
                val_loss = test_metrics['loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                    print(f"New best model saved (loss: {val_loss:.4f})")
                else:
                    self.patience_counter += 1
                    print(f"No improvement for {self.patience_counter} epochs")
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {epoch} epochs")
                        if self.best_model_state is not None:
                            self.model.load_state_dict(self.best_model_state)
                            print("Restored best model")
                        break
        
        return dict(metrics)


def data_preprocessing(data_path='C:/Users/lelik/mine/e-commerce-recommender/data', max_sessions=10000):
    """
    Preprocess the e-commerce data for the recommender model.
    
    Args:
        data_path: Path to the data directory
        max_sessions: Maximum number of sessions to process
        
    Returns:
        Dictionary of preprocessed data
    """
    import pandas as pd
    from pathlib import Path
    import os
    
    print("Loading and preprocessing data...")
    
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_path} does not exist")
    
    events_path = data_dir / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Events file {events_path} does not exist")
    
    try:
        events_chunks = pd.read_csv(events_path, chunksize=100000)
    except Exception as e:
        print(f"Error reading CSV with default engine: {e}. Trying 'python' engine.")
        try:
            events_chunks = pd.read_csv(events_path, chunksize=100000, engine='python')
        except Exception as e_py:
            print(f"Error reading CSV with python engine: {e_py}")
            raise
    
    user_sessions = defaultdict(list)
    item_to_category = {}
    
    category_path = data_dir / "category_tree.csv"
    try:
        category_data = pd.read_csv(category_path)
        category_hierarchy = dict(zip(category_data['categoryid'], category_data['parentid']))
    except Exception as e:
        print(f"Warning: Could not load category tree data: {e}. Proceeding without hierarchy.")
        category_hierarchy = {}
    
    event_to_id = {'view': 1, 'addtocart': 2, 'transaction': 3}
    
    # Consider loading properties more robustly or skipping if optional
    props_path1 = data_dir / "item_properties_part1.csv"
    try:
        # Load only necessary columns and process in chunks if large
        props_iter = pd.read_csv(props_path1, usecols=['itemid', 'property', 'value'], chunksize=500000)
        for props_chunk in props_iter:
             category_items = props_chunk[props_chunk['property'] == 'categoryid']
             for _, row in category_items.iterrows():
                 try:
                     item_to_category[int(row['itemid'])] = int(row['value'])
                 except (ValueError, TypeError):
                     continue # Skip malformed item/category IDs
    except FileNotFoundError:
         print(f"Warning: Item properties file not found at {props_path1}. Categories might be missing.")
    except Exception as e:
        print(f"Error loading item properties: {e}. Categories might be incomplete.")


    print("Processing events data...")
    session_count = 0
    processed_users = set()

    try:
        for chunk in events_chunks:
            chunk = chunk.sort_values(['visitorid', 'timestamp'])
            
            for user_id, user_events in chunk.groupby('visitorid'):
                 if user_id in processed_users: # Avoid reprocessing users across chunks if possible
                     continue # This assumes chunks might split user sessions, adjust if needed

                 user_events = user_events.sort_values('timestamp')
                 current_session_items = []
                 current_session_categories = []
                 current_session_actions = []
                 last_timestamp = None
                
                 for _, event in user_events.iterrows():
                    item_id = event['itemid']
                    timestamp = event['timestamp'] # Assuming ms
                    action_id = event_to_id.get(event['event'], 0) # Use 0 for unknown events
                    category_id = item_to_category.get(item_id, 0) # Use 0 for unknown categories

                    session_timeout_ms = 30 * 60 * 1000 # 30 minutes

                    if last_timestamp is not None and (timestamp - last_timestamp) > session_timeout_ms:
                        if len(current_session_items) >= 2:
                            session_key = f"{user_id}_{session_count}"
                            user_sessions[session_key] = {
                                'items': current_session_items.copy(),
                                'categories': current_session_categories.copy(),
                                'actions': current_session_actions.copy()
                            }
                            session_count += 1
                            if session_count % 1000 == 0:
                                print(f"Processed {session_count} sessions...")
                            if session_count >= max_sessions:
                                raise StopIteration # Stop processing chunks

                        current_session_items = []
                        current_session_categories = []
                        current_session_actions = []

                    current_session_items.append(item_id)
                    current_session_categories.append(category_id)
                    current_session_actions.append(action_id)
                    last_timestamp = timestamp
                
                 # Add the last session for the user
                 if len(current_session_items) >= 2:
                    session_key = f"{user_id}_{session_count}"
                    user_sessions[session_key] = {
                         'items': current_session_items,
                         'categories': current_session_categories,
                         'actions': current_session_actions
                    }
                    session_count += 1
                    if session_count >= max_sessions:
                         raise StopIteration

                 processed_users.add(user_id) # Mark user as processed for this run


            if session_count >= max_sessions:
                break
                
    except StopIteration:
        print(f"Stopped after reaching {max_sessions} sessions.")
    except Exception as e:
        print(f"Error during event processing: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        # Decide whether to raise or return potentially incomplete data
        # raise e 
        
    if len(user_sessions) < 10: # Check if enough data was gathered
        print(f"Warning: Only {len(user_sessions)} sessions found. Model training might be suboptimal.")
        if not user_sessions:
             raise ValueError("No valid sessions found for training.") # Cannot proceed without data

    print(f"Total sessions extracted: {len(user_sessions)}")
    
    item_sequences = [session['items'] for session in user_sessions.values()]
    category_sequences = [session['categories'] for session in user_sessions.values()]
    action_sequences = [session['actions'] for session in user_sessions.values()]
    
    return {
        'item_sequences': item_sequences,
        'category_sequences': category_sequences,
        'action_sequences': action_sequences,
        'category_hierarchy': category_hierarchy,
        'item_to_category': item_to_category
    }


def create_data_loaders(data_splits, batch_size=32, sequence_length=5):
    """
    Create PyTorch DataLoaders from data splits.
    
    Args:
        data_splits: Dictionary of data splits from prepare_data
        batch_size: Batch size for DataLoaders
        sequence_length: Sequence length for training samples
        
    Returns:
        Dictionary of DataLoaders
    """
    from torch.utils.data import DataLoader, Dataset
    import torch
    import numpy as np
    
    class SequenceDataset(Dataset):
        def __init__(self, item_sequences, category_sequences, action_sequences, sequence_length):
            self.samples = []
            min_seq_len = sequence_length + 1 # Need input + target

            for i in range(len(item_sequences)):
                item_seq = item_sequences[i]
                category_seq = category_sequences[i]
                action_seq = action_sequences[i]
                
                if len(item_seq) < min_seq_len:
                    continue
                
                # Ensure all sequences have the same length for consistency check
                if not (len(item_seq) == len(category_seq) == len(action_seq)):
                     print(f"Warning: Skipping sequence {i} due to length mismatch.")
                     continue

                for j in range(len(item_seq) - sequence_length):
                    item_input = item_seq[j : j + sequence_length]
                    category_input = category_seq[j : j + sequence_length]
                    action_input = action_seq[j : j + sequence_length]
                    
                    item_target = item_seq[j + sequence_length]
                    category_target = category_seq[j + sequence_length]
                    action_target = action_seq[j + sequence_length]
                    
                    # Basic validation (e.g., check if indices are within expected range)
                    # Add more checks if necessary based on data characteristics

                    self.samples.append({
                        'item_input': item_input,
                        'category_input': category_input,
                        'action_input': action_input,
                        'item_target': item_target,
                        'category_target': category_target,
                        'action_target': action_target
                    })
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            return {
                'item_input': torch.tensor(sample['item_input'], dtype=torch.long),
                'category_input': torch.tensor(sample['category_input'], dtype=torch.long),
                'action_input': torch.tensor(sample['action_input'], dtype=torch.long),
                'item_target': torch.tensor(sample['item_target'], dtype=torch.long),
                'category_target': torch.tensor(sample['category_target'], dtype=torch.long),
                'action_target': torch.tensor(sample['action_target'], dtype=torch.long)
            }
    
    try:
         train_dataset = SequenceDataset(
             data_splits['train']['items'],
             data_splits['train']['categories'],
             data_splits['train']['actions'],
             sequence_length
         )
         test_dataset = SequenceDataset(
             data_splits['test']['items'],
             data_splits['test']['categories'],
             data_splits['test']['actions'],
             sequence_length
         )
         eval_dataset = SequenceDataset(
             data_splits['eval']['items'],
             data_splits['eval']['categories'],
             data_splits['eval']['actions'],
             sequence_length
         )

         print(f"Created datasets: Train={len(train_dataset)}, Test={len(test_dataset)}, Eval={len(eval_dataset)}")

         if len(train_dataset) == 0:
              raise ValueError("Training dataset is empty. Check sequence_length and data preprocessing.")

    except KeyError as e:
         print(f"Error creating dataset: Missing key {e} in data_splits. Ensure prepare_data returns correct structure.")
         raise
    except Exception as e:
         print(f"An error occurred during dataset creation: {e}")
         raise


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for debugging
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=0)
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'eval_loader': eval_loader
    }


def run_enhanced_training_pipeline(
    data_path='C:/Users/lelik/mine/e-commerce-recommender/data', # Make data path an argument
    max_sessions=3000,
    batch_size=64,
    sequence_length=10,
    num_epochs=10,
    embedding_dim=128,
    lstm_hidden_dim=128,
    learning_rate=0.001,
    weight_decay=0.01,
    dropout_rate=0.3,
    use_weighted_loss=True,
    use_candidate_sampling=True,
    candidate_sample_size=1000,
    early_stopping=True,
    device=None # Allow device selection
):
    """
    Run an enhanced training pipeline with improved metrics and techniques.
    """
    import torch
    import random
    import numpy as np
    
    if device is None:
         device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("Step 1: Preprocessing data...")
    data = data_preprocessing(data_path=data_path, max_sessions=max_sessions)
    
    print("Step 2: Analyzing data and preparing model inputs...")
    items_by_category = defaultdict(list)
    all_items = set()
    all_categories = set()
    all_actions = set()

    for i in range(len(data['item_sequences'])):
        for j in range(len(data['item_sequences'][i])):
            item_id = data['item_sequences'][i][j]
            category_id = data['category_sequences'][i][j]
            action_id = data['action_sequences'][i][j]

            all_items.add(item_id)
            all_categories.add(category_id)
            all_actions.add(action_id)

            if item_id not in items_by_category[category_id]:
                items_by_category[category_id].append(item_id)
    
    num_items = max(all_items) + 1 if all_items else 1
    num_categories = max(all_categories) + 1 if all_categories else 1
    num_actions = max(all_actions) + 1 if all_actions else 1


    print(f"  Number of unique items found: {num_items-1}")
    print(f"  Number of unique categories found: {num_categories-1}")
    print(f"  Number of unique actions found: {num_actions-1}")

    category_counts = {cat: len(items) for cat, items in items_by_category.items() if cat != 0} # Exclude padding category
    if category_counts:
        print(f"  Avg items per category: {np.mean(list(category_counts.values())):.2f}")
        print(f"  Max items per category: {max(category_counts.values())}")
    else:
        print("  No category information found or processed.")

    
    print("Step 3: Initializing enhanced model...")
    model = ECommerceRecommenderModel(
        num_items=num_items,
        num_categories=num_categories,
        num_actions=num_actions,
        embedding_dim=embedding_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        markov_order=2, # Consider making this a parameter
        dropout_rate=dropout_rate,
        max_items_for_markov=min(10000, num_items // 2), # Heuristic, adjust if needed
        use_candidate_sampling=use_candidate_sampling,
        candidate_sample_size=candidate_sample_size,
        items_by_category=dict(items_by_category) # Convert back to dict
    )
    
    print("Step 4: Building item relationships (co-occurrence, Markov)...")
    # Markov transitions require sequences; ensure they are suitable
    # Consider if Markov update should happen only on train split
    model.update_markov_transitions(data['item_sequences'], entity_type='item')
    model.update_markov_transitions(data['category_sequences'], entity_type='category')
    model.update_markov_transitions(data['action_sequences'], entity_type='action')
    model.update_item_co_occurrence(data['item_sequences'])
    # update_items_by_category is done during model init now

    print(f"  Item co-occurrence pairs calculated.")
    print(f"  Markov transitions updated.")

    
    print("Step 5: Setting up trainer...")
    trainer = ECommerceRecommenderTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_weighted_loss=use_weighted_loss,
        device=device # Pass the selected device
    )
    
    print("Step 6: Preparing data splits...")
    # Consider using stratified split if data is highly imbalanced
    data_splits = trainer.prepare_data(
        data['item_sequences'],
        data['category_sequences'],
        data['action_sequences']
        # Add stratification option here if needed
    )
    
    print(f"  Training sequences: {len(data_splits['train']['items']):,}")
    print(f"  Testing sequences: {len(data_splits['test']['items']):,}")
    print(f"  Evaluation sequences: {len(data_splits['eval']['items']):,}")
    
    print("Step 7: Creating data loaders...")
    loaders = create_data_loaders(
        data_splits,
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    print("Step 8: Training model...")
    metrics = trainer.train(
        loaders['train_loader'],
        loaders['test_loader'],
        loaders['eval_loader'],
        num_epochs=num_epochs,
        early_stopping=early_stopping,
        log_interval=max(1, len(loaders['train_loader']) // 10) # Adjust log interval
    )
    
    print("Training complete!")
    
    print("Final Evaluation Metrics (from last epoch):")
    if metrics:
         last_epoch_metrics = {k: v[-1] for k, v in metrics.items() if v}
         print(', '.join([f'{k}: {v:.4f}' for k, v in last_epoch_metrics.items()]))
    else:
         print("Metrics dictionary is empty.")

    
    return model, metrics, data


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description="E-Commerce Recommender System Training")
    parser.add_argument('--data_path', type=str, default='C:/Users/lelik/mine/e-commerce-recommender/data', help='Path to the data directory')
    parser.add_argument('--max_sessions', type=int, default=500, help='Maximum number of sessions to process')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length for LSTM input')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of embeddings')
    parser.add_argument('--lstm_hidden_dim', type=int, default=128, help='Hidden dimension of LSTM')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--no_weighted_loss', action='store_true', help='Disable weighted loss')
    parser.add_argument('--no_candidate_sampling', action='store_true', help='Disable candidate sampling')
    parser.add_argument('--candidate_sample_size', type=int, default=1000, help='Number of candidates for sampling')
    parser.add_argument('--no_early_stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., "cpu", "cuda")')

    args = parser.parse_args()

    print("E-Commerce Recommender System - Training Pipeline")
    print("================================================")
    print(f"Running with parameters: {vars(args)}")

    
    try:
        model, metrics, data = run_enhanced_training_pipeline(
            data_path=args.data_path,
            max_sessions=args.max_sessions,        
            batch_size=args.batch_size,           
            sequence_length=args.sequence_length, 
            num_epochs=args.num_epochs,            
            embedding_dim=args.embedding_dim,       
            lstm_hidden_dim=args.lstm_hidden_dim,     
            learning_rate=args.lr,     
            weight_decay=args.weight_decay,       
            dropout_rate=args.dropout_rate,        
            use_weighted_loss=not args.no_weighted_loss, 
            use_candidate_sampling=not args.no_candidate_sampling, 
            candidate_sample_size=args.candidate_sample_size,   
            early_stopping=not args.no_early_stopping,
            device=args.device      
        )
        
        # Optionally save the model or results here
        # torch.save(model.state_dict(), 'final_model.pth')
        # print("Model state saved to final_model.pth")

    except FileNotFoundError as e:
         print(f"Error: Data file not found. {e}")
         print("Please ensure the data path is correct and the necessary CSV files exist.")
    except ValueError as e:
         print(f"Error: {e}")
         print("This might be due to insufficient data or issues during preprocessing/dataloader creation.")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred during the training pipeline:")
        print(traceback.format_exc())

    finally:
        print("Script execution finished.")


if __name__ == "__main__":
    print("E-Commerce Recommender System")
    print("============================")
    print("1. Running enhanced training pipeline with improved metrics")
    print("   This will process more sessions and use advanced techniques.")
    
    try:
        model, metrics, data = run_enhanced_training_pipeline(
            max_sessions=1000,        # Process 500 sessions
            batch_size=64,           # Batch size
            sequence_length=10,      # Use 10 interactions to predict the next
            num_epochs=5,            # Train for 5 epochs
            embedding_dim=128,       # Embedding dimension
            lstm_hidden_dim=128,     # LSTM hidden dimension
            learning_rate=0.001,     # Learning rate
            weight_decay=0.01,       # Weight decay
            dropout_rate=0.3,        # Dropout rate
            use_weighted_loss=True,  # Use weighted loss functions
            use_candidate_sampling=True,  # Use candidate sampling
            candidate_sample_size=1000,   # Candidate sample size for real data
            early_stopping=True      # Use early stopping
        )
        
    except Exception as e:
        import traceback
        print(f"\nAn error occurred during training:")
        print(traceback.format_exc())
        print("\nYou can run the standard pipeline with:")
        print("model, metrics, data = run_training_pipeline(max_sessions=3000)") 

    finally:
        print("\nDebug: Script execution completed.") 