import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
import gc

class HierarchicalMarkovModel(nn.Module):
    """
    Hierarchical Markov Model for e-commerce recommendations using PyTorch.
    
    Models the probability of the next item by decomposing the problem:
    P(next_item) = P(next_category) * P(next_subcategory | next_category) * P(next_item | next_subcategory)
    
    Incorporates item properties and session context into the prediction.
    """
    
    def __init__(self, embedding_dim=64, alpha=0.1, device='cpu'):
        """
        Initialize the PyTorch Hierarchical Markov Model.
        
        Args:
            embedding_dim: Dimension for embedding representations
            alpha: Smoothing parameter for transition probabilities
            device: PyTorch device for computation
        """
        super(HierarchicalMarkovModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.device = device
        
        # Category hierarchy mappings
        self.category_levels = {}        # category_id -> level
        self.parent_categories = {}      # category_id -> parent_id
        self.category_children = defaultdict(set)  # parent_id -> set of child_ids
        
        # Item mappings
        self.item_to_category = {}       # item_id -> category_id
        self.category_items = defaultdict(set)  # category_id -> set of item_ids
        
        # Encoder/decoder mappings
        self.category_encoder = None
        self.item_encoder = None
        self.property_encoders = {}
        
        # Transition matrices (will be PyTorch tensors)
        self.category_transitions = None
        self.subcategory_transitions = {}
        self.item_transitions = None
        
        # Item properties
        self.item_properties = defaultdict(dict)
        
        # PyTorch embedding layers
        self.category_embeddings = None
        self.item_embeddings = None
        self.property_embeddings = {}
        
        # Network layers for transition prediction
        self.category_predictor = None
        self.subcategory_predictor = None
        self.item_predictor = None
        
        # Session context
        self.context_encoder = None
        
        print(f"Initialized Hierarchical Markov Model using device: {device}")
    
    def fit(self, events_df, category_df, item_properties_df, batch_size=1024, n_epochs=5, 
           max_events=None, max_items=None, max_categories=None):
        """
        Fit the hierarchical model to the data.
        
        Args:
            events_df: DataFrame with user events (user_id, item_id, event, timestamp)
            category_df: DataFrame with category hierarchy (categoryid, parentid, level)
            item_properties_df: DataFrame with item properties (itemid, property, value)
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            max_events: Maximum number of events to use (None = use all)
            max_items: Maximum number of unique items to include (None = use all)
            max_categories: Maximum number of categories to include (None = use all)
        """
        print("Building hierarchical Markov model with PyTorch...")
        
        # Sample dataset if limits are specified
        if max_events or max_items or max_categories:
            events_df, category_df, item_properties_df = self.sample_dataset(
                events_df, category_df, item_properties_df,
                max_events=max_events, max_items=max_items, max_categories=max_categories
            )
        
        # Process category hierarchy
        self._process_category_hierarchy(category_df)
        
        # Process item properties
        self._process_item_properties(item_properties_df)
        
        # Process events and create sessions
        sessions = self._identify_sessions(events_df)
        
        # Build transition tensors
        self._build_transition_tensors(sessions)
        
        # Initialize embedding layers
        self._init_embedding_layers()
        
        # Train the model
        self._train_model(sessions, batch_size, n_epochs)
        
        print("Hierarchical Markov model trained successfully")
        return self
    
    def _process_category_hierarchy(self, category_df):
        """Process the category hierarchy for a two-level structure (parent-child only)."""
        print("Processing two-level category hierarchy...")
        
        # First identify the actual column names based on what's available
        cat_col = next((col for col in ['categoryid', 'category_id'] if col in category_df.columns), None)
        parent_col = next((col for col in ['parentid', 'parent_id'] if col in category_df.columns), None)
        
        if not cat_col:
            raise ValueError(f"Could not find category ID column in category dataframe. Available columns: {category_df.columns.tolist()}")
        
        # Create category encoder
        self.category_encoder = LabelEncoder()
        category_ids = category_df[cat_col].tolist()
        self.category_encoder.fit(category_ids)
        
        # Get encoded categories
        encoded_categories = self.category_encoder.transform(category_ids)
        
        # In a two-level hierarchy:
        # Level 0: Root categories (no parent)
        # Level 1: Child categories (has parent)
        root_categories = []
        child_categories = []
        
        # Process parent-child relationships
        for i, row in category_df.iterrows():
            cat_id = row[cat_col]
            
            # Handle case where parent column doesn't exist in sampled data
            if parent_col is None:
                # Assume all are root categories if no parent column
                root_categories.append(cat_id)
                self.category_levels[cat_id] = 0
                continue
            
            parent_id = row[parent_col]
            
            if pd.isna(parent_id):
                # This is a root category (level 0)
                root_categories.append(cat_id)
                self.category_levels[cat_id] = 0
            else:
                # This is a child category (level 1)
                child_categories.append(cat_id)
                self.category_levels[cat_id] = 1
                
                # Store parent relationship
                self.parent_categories[cat_id] = parent_id
                self.category_children[parent_id].add(cat_id)
        
        # For categories whose parents aren't in the sampled data, treat them as root
        if parent_col:
            for cat_id, parent_id in self.parent_categories.items():
                if parent_id not in self.category_levels:
                    # Parent not in dataset, so treat this as a root category
                    self.category_levels[cat_id] = 0
                    root_categories.append(cat_id)
                    # Remove from child list
                    if cat_id in child_categories:
                        child_categories.remove(cat_id)
        
        # Create tensors for category hierarchy
        n_categories = len(self.category_encoder.classes_)
        
        # Create level tensor - maps each category to its level (0 or 1)
        self.level_tensor = torch.zeros(n_categories, dtype=torch.long, device=self.device)
        for cat_id, level in self.category_levels.items():
            try:
                cat_idx = self.category_encoder.transform([cat_id])[0]
                self.level_tensor[cat_idx] = level
            except:
                pass  # Skip if category can't be transformed
        
        # Create parent tensor - maps each category to its parent
        self.parent_tensor = torch.full((n_categories,), -1, dtype=torch.long, device=self.device)
        for cat_id, parent_id in self.parent_categories.items():
            try:
                if parent_id in self.category_encoder.classes_:
                    cat_idx = self.category_encoder.transform([cat_id])[0]
                    parent_idx = self.category_encoder.transform([parent_id])[0]
                    self.parent_tensor[cat_idx] = parent_idx
            except:
                pass  # Skip if category or parent can't be transformed
        
        print(f"Processed {n_categories} categories")
        print(f"Root categories: {len(root_categories)}")
        print(f"Child categories: {len(child_categories)}")
    
    def _process_item_properties(self, item_properties_df):
        """Process item properties and create PyTorch-friendly structures."""
        print("Processing item properties...")
        
        # Select relevant properties for the model
        relevant_properties = ['categoryid', 'brand', 'price', 'color', 'size', 'available']
        filtered_props = item_properties_df[item_properties_df['property'].isin(relevant_properties)]
        
        # Item encoder
        unique_items = filtered_props['itemid'].unique()
        self.item_encoder = LabelEncoder()
        self.item_encoder.fit(unique_items)
        
        # Create property encoders
        self.property_encoders = {}
        for prop in relevant_properties:
            if prop != 'price':  # Don't encode numeric properties
                prop_values = filtered_props[filtered_props['property'] == prop]['value'].dropna().unique()
                if len(prop_values) > 0:
                    self.property_encoders[prop] = LabelEncoder()
                    self.property_encoders[prop].fit(prop_values)
        
        # Create item-category mappings more efficiently using vectorized operations
        category_rows = filtered_props[filtered_props['property'] == 'categoryid']
        item_categories = dict(zip(category_rows['itemid'], category_rows['value']))
        self.item_to_category = item_categories
        
        # Create category_items mapping
        for item_id, category_id in item_categories.items():
            self.category_items[category_id].add(item_id)
        
        # Process other properties more efficiently
        grouped_props = filtered_props.groupby('itemid')
        for item_id, group in grouped_props:
            prop_dict = dict(zip(group['property'], group['value']))
            self.item_properties[item_id] = prop_dict
        
        # Free up memory
        del filtered_props
        del grouped_props
        gc.collect()
        
        # Create item-category tensor
        n_items = len(self.item_encoder.classes_)
        n_categories = len(self.category_encoder.classes_)
        
        # Initialize tensor with -1 (unknown category)
        self.item_category_tensor = torch.full((n_items,), -1, dtype=torch.long, device=self.device)
        
        # Fill tensor with encoded categories using batched processing
        batch_size = 10000
        item_ids = list(self.item_to_category.keys())
        
        for i in range(0, len(item_ids), batch_size):
            batch_items = item_ids[i:i+batch_size]
            valid_items = [item for item in batch_items if item in self.item_encoder.classes_]
            valid_categories = [self.item_to_category[item] for item in valid_items]
            
            # Filter categories that are in the encoder
            valid_items_with_cats = [
                (item, cat) for item, cat in zip(valid_items, valid_categories) 
                if cat in self.category_encoder.classes_
            ]
            
            if valid_items_with_cats:
                batch_items, batch_cats = zip(*valid_items_with_cats)
                item_indices = self.item_encoder.transform(batch_items)
                cat_indices = self.category_encoder.transform(batch_cats)
                
                for item_idx, cat_idx in zip(item_indices, cat_indices):
                    self.item_category_tensor[item_idx] = cat_idx
        
        # Create property tensors with batched processing
        self.property_tensors = {}
        for prop, encoder in self.property_encoders.items():
            # For each property, create a tensor mapping items to property values
            self.property_tensors[prop] = torch.full(
                (n_items,), -1, dtype=torch.long, device=self.device
            )
            
            # Collect all items with this property
            items_with_prop = []
            prop_values = []
            
            for item_id, props in self.item_properties.items():
                if prop in props and item_id in self.item_encoder.classes_:
                    prop_value = props[prop]
                    if prop_value in encoder.classes_:
                        items_with_prop.append(item_id)
                        prop_values.append(prop_value)
            
            # Process in batches
            for i in range(0, len(items_with_prop), batch_size):
                batch_items = items_with_prop[i:i+batch_size]
                batch_values = prop_values[i:i+batch_size]
                
                try:
                    item_indices = self.item_encoder.transform(batch_items)
                    prop_indices = encoder.transform(batch_values)
                    
                    for item_idx, prop_idx in zip(item_indices, prop_indices):
                        self.property_tensors[prop][item_idx] = prop_idx
                except:
                    # Skip any errors in encoding
                    pass
            
            # Free memory
            gc.collect()
        
        print(f"Processed {len(self.item_properties)} items with properties")
        print(f"Mapped {len(self.item_to_category)} items to {len(self.category_items)} categories")
    
    def _identify_sessions(self, events_df, session_minutes=30):
        """Identify user sessions and return a dictionary of session data."""
        print("Identifying user sessions...")
        
        # Check and rename columns if needed
        column_mapping = {
            'visitorid': 'user_id',
            'itemid': 'item_id',
            'timestamp': 'datetime'
        }
        
        # Rename columns based on what exists in the dataframe
        events_df = events_df.rename(columns=column_mapping)
        
        events_df['datetime'] = pd.to_datetime(events_df['datetime'])
        
        # Sort by user and time
        events_df = events_df.sort_values(['user_id', 'datetime'])
        
        # Calculate time difference between consecutive events
        events_df['time_diff'] = events_df.groupby('user_id')['datetime'].diff()
        
        # Start new session when time gap > session_minutes
        session_threshold = pd.Timedelta(minutes=session_minutes)
        # Use a safer comparison approach
        events_df['new_session'] = events_df['time_diff'].isna() | (events_df['time_diff'].dt.total_seconds() > session_threshold.total_seconds())
        
        # Create global session ID
        events_df['global_session_id'] = events_df['user_id'].astype(str) + "_" + events_df['new_session'].astype(str)
        
        # Group by session
        sessions = {}
        for session_id, session_data in events_df.groupby('global_session_id'):
            sessions[session_id] = session_data.sort_values('datetime')
        
        print(f"Identified {len(sessions)} user sessions")
        return sessions
    
    def _build_transition_tensors(self, sessions):
        """Build transition tensors at each hierarchy level using PyTorch."""
        print("Building transition tensors...")
        
        n_categories = len(self.category_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        # Initialize transition count tensors
        self.category_transitions = torch.zeros(
            (n_categories, n_categories), 
            dtype=torch.float, 
            device=self.device
        )
        
        self.item_transitions = torch.zeros(
            (n_items, n_items), 
            dtype=torch.float, 
            device=self.device
        )
        
        # Initialize subcategory transitions (for each parent category)
        for parent_id, children in self.category_children.items():
            if parent_id in self.category_encoder.classes_:
                n_children = len(children)
                if n_children > 1:
                    # Only create transitions for parents with multiple children
                    parent_idx = self.category_encoder.transform([parent_id])[0]
                    child_indices = [
                        self.category_encoder.transform([child])[0] 
                        for child in children 
                        if child in self.category_encoder.classes_
                    ]
                    
                    # Create transition matrix for this parent's children
                    if len(child_indices) > 1:
                        size = max(child_indices) + 1
                        self.subcategory_transitions[parent_idx] = torch.zeros(
                            (size, size), 
                            dtype=torch.float, 
                            device=self.device
                        )
        
        # Process sessions to build transition counts
        transition_count = 0
        for session_id, session_data in tqdm(sessions.items()):
            items = session_data['item_id'].tolist()
            
            # Skip short sessions
            if len(items) < 2:
                continue
            
            # Get encoded items that exist in our encoders
            valid_items = [
                item for item in items 
                if item in self.item_encoder.classes_ and item in self.item_to_category
            ]
            
            if len(valid_items) < 2:
                continue
                
            encoded_items = self.item_encoder.transform(valid_items)
            
            # Process transitions
            for i in range(len(encoded_items) - 1):
                item1_idx, item2_idx = encoded_items[i], encoded_items[i+1]
                item1, item2 = valid_items[i], valid_items[i+1]
                
                # Update item transition counts
                self.item_transitions[item1_idx, item2_idx] += 1
                
                # Get categories for these items
                if item1 in self.item_to_category and item2 in self.item_to_category:
                    cat1 = self.item_to_category[item1]
                    cat2 = self.item_to_category[item2]
                    
                    if cat1 in self.category_encoder.classes_ and cat2 in self.category_encoder.classes_:
                        cat1_idx = self.category_encoder.transform([cat1])[0]
                        cat2_idx = self.category_encoder.transform([cat2])[0]
                        
                        # Update category transition counts
                        self.category_transitions[cat1_idx, cat2_idx] += 1
                        
                        # Update subcategory transition counts if they have the same parent
                        if (cat1 in self.parent_categories and 
                            cat2 in self.parent_categories and 
                            self.parent_categories[cat1] == self.parent_categories[cat2]):
                            
                            parent = self.parent_categories[cat1]
                            if parent in self.category_encoder.classes_:
                                parent_idx = self.category_encoder.transform([parent])[0]
                                if parent_idx in self.subcategory_transitions:
                                    self.subcategory_transitions[parent_idx][cat1_idx, cat2_idx] += 1
                
                transition_count += 1
                
                # Periodically free memory
                if transition_count % 500000 == 0:
                    gc.collect()
        
        # Normalize transition matrices (add smoothing and convert to probabilities)
        self._normalize_transitions()
        
        print(f"Built transition tensors from {transition_count} transitions")
    
    def _normalize_transitions(self):
        """Apply smoothing and normalize transition matrices to get probabilities."""
        # Normalize category transitions
        self.category_transitions = self._normalize_tensor(self.category_transitions)
        
        # Normalize subcategory transitions
        for parent_idx, transitions in self.subcategory_transitions.items():
            self.subcategory_transitions[parent_idx] = self._normalize_tensor(transitions)
        
        # Normalize item transitions
        self.item_transitions = self._normalize_tensor(self.item_transitions)
    
    def _normalize_tensor(self, transitions):
        """Apply smoothing and normalize a transition tensor"""
        row_sums = transitions.sum(dim=1, keepdim=True)
        mask = row_sums == 0
        row_sums[mask] = 1  # Avoid division by zero
        return (transitions + self.alpha) / (row_sums + self.alpha * transitions.shape[1])
    
    def _init_embedding_layers(self):
        """Initialize PyTorch embedding layers."""
        n_categories = len(self.category_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        # Category embeddings
        self.category_embeddings = nn.Embedding(
            n_categories, 
            self.embedding_dim,
            device=self.device
        )
        
        # Item embeddings
        self.item_embeddings = nn.Embedding(
            n_items,
            self.embedding_dim,
            device=self.device
        )
        
        # Property embeddings
        for prop, encoder in self.property_encoders.items():
            n_values = len(encoder.classes_)
            self.property_embeddings[prop] = nn.Embedding(
                n_values + 1,  # +1 for unknown
                self.embedding_dim // 2,
                padding_idx=n_values,
                device=self.device
            )
        
        # Session context encoder (LSTM)
        self.context_encoder = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
            device=self.device
        )
        
        # Prediction networks
        self.category_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, n_categories),
            nn.Softmax(dim=1)
        ).to(self.device)
        
        self.item_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def _train_model(self, sessions, batch_size=1024, n_epochs=5):
        """Train the neural components of the model."""
        print("Training neural components...")
        
        # Create dataset from sessions
        session_data = []
        for session_id, session_df in sessions.items():
            items = session_df['item_id'].tolist()
            if len(items) >= 3:  # Need at least 3 items for training
                valid_items = [
                    item for item in items 
                    if item in self.item_encoder.classes_ and item in self.item_to_category
                ]
                if len(valid_items) >= 3:
                    encoded_items = self.item_encoder.transform(valid_items).tolist()
                    session_data.append(encoded_items)
        
        # Free memory
        gc.collect()
        
        # Create data loader
        dataset = SessionDataset(session_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=session_collate_fn
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # Training loop
        self.train()
        for epoch in range(n_epochs):
            total_loss = 0
            
            for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
                # Get context and target
                context_items, target_items = batch
                
                # Convert to device - ensure all tensors are on the correct device
                context_items = [torch.tensor(session, dtype=torch.long, device=self.device) 
                               for session in context_items]
                target_items = torch.tensor(target_items, dtype=torch.long, device=self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                batch_loss = 0
                for i, (context, target) in enumerate(zip(context_items, target_items)):
                    # Check for empty context
                    if len(context) == 0:
                        continue
                        
                    # Get item embeddings for context
                    item_embeds = self.item_embeddings(context)
                    
                    # Encode context safely
                    context_packed = nn.utils.rnn.pack_sequence([item_embeds])
                    _, (hidden, _) = self.context_encoder(context_packed)
                    # Safe squeezing with dimension check
                    context_encoding = hidden[0] if hidden.size(0) == 1 else hidden.squeeze(0)
                    
                    # Get target item embedding
                    target_embed = self.item_embeddings(target)
                    
                    # Get target category with error handling
                    target_category = self.item_category_tensor[target]
                    if target_category == -1:
                        # Skip items with unknown category
                        continue
                    
                    # Category prediction loss
                    cat_pred_input = torch.cat([context_encoding, target_embed], dim=0).unsqueeze(0)
                    cat_pred = self.category_predictor(cat_pred_input)
                    cat_loss = F.cross_entropy(cat_pred, target_category.unsqueeze(0))
                    
                    # Item prediction loss (treat as binary classification)
                    # For each target item, predict if it follows the context
                    item_pred_input = torch.cat([
                        context_encoding, 
                        target_embed, 
                        self.category_embeddings(target_category)
                    ], dim=0).unsqueeze(0)
                    
                    item_pred = self.item_predictor(item_pred_input)
                    item_loss = F.binary_cross_entropy(item_pred, torch.ones(1, 1, device=self.device))
                    
                    # Combined loss
                    batch_loss += cat_loss + item_loss
                
                # Check if we had any valid examples
                if batch_loss > 0:
                    # Average loss for the batch
                    batch_loss /= len(target_items)
                    
                    # Backward pass
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item()
            
            # Print epoch results
            avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
            
            # Free memory between epochs
            gc.collect()
    
    def predict_next_items(self, session_items, k=10):
        """
        Predict the top-k next items given the current session items.
        
        Args:
            session_items: List of item IDs in the current session
            k: Number of recommendations to generate
            
        Returns:
            List of recommended item IDs
        """
        self.eval()
        
        try:
            with torch.no_grad():
                # Filter valid items
                valid_items = [
                    item for item in session_items 
                    if item in self.item_encoder.classes_ and item in self.item_to_category
                ]
                
                if not valid_items:
                    print("Warning: No valid items in session")
                    return []
                
                # Need at least 2 items for meaningful prediction
                if len(valid_items) < 2:
                    print(f"Warning: Only {len(valid_items)} valid items in session, need at least 2 for reliable predictions")
                    # Add a simple fallback prediction for very short sessions
                    item = valid_items[0]
                    cat_id = self.item_to_category[item]
                    # Return random items from same category
                    same_cat_items = list(self.category_items.get(cat_id, []))
                    same_cat_items = [i for i in same_cat_items if i != item and i in self.item_encoder.classes_]
                    if same_cat_items and len(same_cat_items) >= k:
                        return np.random.choice(same_cat_items, min(k, len(same_cat_items)), replace=False).tolist()
                    else:
                        return []
                
                # Get encoded items
                encoded_items = self.item_encoder.transform(valid_items)
                item_tensor = torch.tensor(encoded_items, dtype=torch.long, device=self.device)
                
                # Get the most recent items (up to last 5)
                recent_items = item_tensor[-5:] if len(item_tensor) > 5 else item_tensor
                
                # Get embeddings for recent items
                item_embeds = self.item_embeddings(recent_items)
                
                # Encode session context
                context_packed = nn.utils.rnn.pack_sequence([item_embeds])
                _, (hidden, _) = self.context_encoder(context_packed)
                # Safe squeezing with dimension check
                context_encoding = hidden[0] if hidden.size(0) == 1 else hidden.squeeze(0)
                
                # Get last item and its category
                last_item = encoded_items[-1]
                last_category = self.item_category_tensor[last_item]
                
                # First predict the next category
                cat_input = torch.cat([
                    context_encoding, 
                    self.item_embeddings(torch.tensor(last_item, device=self.device))
                ], dim=0).unsqueeze(0)
                
                category_probs = self.category_predictor(cat_input).squeeze(0)
                
                # Get top categories (could be multiple)
                top_k_cats = min(3, category_probs.shape[0])  # Consider top 3 categories
                top_cats = torch.topk(category_probs, top_k_cats).indices
                
                # For efficiency, use the Markov transitions for initial filtering
                # Get candidates from last item transitions
                candidate_scores = {}
                
                # 1. Use Markov transitions for initial scoring
                # Get top items based on transitions from last item
                markov_scores = self.item_transitions[last_item].clone()
                
                # 2. Boost scores for items in predicted categories
                for cat_idx in top_cats:
                    cat_id = self.category_encoder.inverse_transform([cat_idx.item()])[0]
                    cat_items = self.category_items.get(cat_id, [])
                    
                    # Convert to indices
                    valid_cat_items = [
                        item for item in cat_items 
                        if item in self.item_encoder.classes_
                    ]
                    
                    if valid_cat_items:
                        cat_items_indices = self.item_encoder.transform(valid_cat_items)
                        
                        # Boost based on category probability
                        for item_idx in cat_items_indices:
                            markov_scores[item_idx] *= (1 + category_probs[cat_idx].item() * 2)
                
                # 3. Filter out items already in session
                markov_scores[item_tensor] = 0
                
                # Get top candidates from Markov model (more than k to allow neural reranking)
                top_candidates = torch.topk(markov_scores, min(k*3, markov_scores.shape[0])).indices
                
                # 4. Neural reranking of candidates
                candidate_scores = torch.zeros(len(top_candidates), device=self.device)
                
                for i, candidate_idx in enumerate(top_candidates):
                    # Get candidate embedding
                    candidate_embed = self.item_embeddings(candidate_idx)
                    
                    # Get candidate category
                    candidate_category = self.item_category_tensor[candidate_idx]
                    if candidate_category == -1:  # Skip items with unknown category
                        continue
                        
                    # Predict score using neural model
                    candidate_input = torch.cat([
                        context_encoding, 
                        candidate_embed,
                        self.category_embeddings(candidate_category)
                    ], dim=0).unsqueeze(0)
                    
                    # Get prediction score
                    candidate_scores[i] = self.item_predictor(candidate_input).item()
                
                # Get final top-k recommendations
                final_scores = candidate_scores
                top_indices = torch.topk(final_scores, min(k, len(top_candidates))).indices
                
                # Convert back to original item IDs
                recommendations = [
                    self.item_encoder.inverse_transform([top_candidates[idx].item()])[0]
                    for idx in top_indices
                ]
                
                return recommendations
        except Exception as e:
            print(f"Error in prediction: {e}")
            return []
    
    def recommend_for_user(self, user_id, user_events, k=10):
        """
        Generate recommendations for a user based on their event history.
        
        Args:
            user_id: User identifier
            user_events: DataFrame with user's events sorted by time
            k: Number of recommendations to generate
            
        Returns:
            List of recommended item IDs
        """
        # Get user's most recent session
        session_items = user_events.sort_values('datetime').tail(10)['item_id'].tolist()
        
        # Generate predictions
        recommendations = self.predict_next_items(session_items, k=k)
        
        return recommendations

    def sample_dataset(self, events_df, category_df, item_properties_df, 
                      max_events=None, max_items=None, max_categories=None, 
                      date_range=None, sample_users=None):
        """
        Sample the dataset to control training time and memory usage.
        
        Args:
            events_df: DataFrame with user events
            category_df: DataFrame with category hierarchy
            item_properties_df: DataFrame with item properties
            max_events: Maximum number of events to use (None = use all)
            max_items: Maximum number of unique items to include (None = use all)
            max_categories: Maximum number of categories to include (None = use all)
            date_range: Tuple of (start_date, end_date) to filter events
            sample_users: Number of users to sample (None = use all)
            
        Returns:
            Tuple of (sampled_events_df, sampled_category_df, sampled_item_properties_df)
        """
        print(f"Sampling dataset with constraints: max_events={max_events}, max_items={max_items}")
        
        # Create copies to avoid modifying originals
        events_df = events_df.copy()
        category_df = category_df.copy()
        item_properties_df = item_properties_df.copy()
        
        # Check if category_df is empty or invalid
        if category_df.empty:
            print("Warning: Category dataframe is empty. Creating a simple one-level hierarchy.")
            # Create a synthetic category dataframe from item properties
            category_prop_rows = item_properties_df[
                (item_properties_df['property'] == 'categoryid') | 
                (item_properties_df['property'] == 'category_id')
            ]
            unique_categories = category_prop_rows['value'].unique()
            
            # Create a simple category dataframe with all categories at level 0
            category_df = pd.DataFrame({
                'categoryid': unique_categories,
                'parentid': [None] * len(unique_categories),
                'level': [0] * len(unique_categories)
            })
        
        # Ensure datetime column is in datetime format
        datetime_col = next((col for col in ['datetime', 'timestamp'] if col in events_df.columns), None)
        if datetime_col:
            events_df[datetime_col] = pd.to_datetime(events_df[datetime_col])
        
        # Filter by date range if specified
        if date_range and datetime_col:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            events_df = events_df[(events_df[datetime_col] >= start_date) & 
                                  (events_df[datetime_col] <= end_date)]
        
        # Sample users if specified
        user_col = next((col for col in ['user_id', 'userid', 'visitorid'] if col in events_df.columns), None)
        if sample_users and user_col:
            unique_users = events_df[user_col].unique()
            if len(unique_users) > sample_users:
                sampled_users = np.random.choice(unique_users, sample_users, replace=False)
                events_df = events_df[events_df[user_col].isin(sampled_users)]
        
        # Sample events if specified
        if max_events and len(events_df) > max_events:
            events_df = events_df.sample(max_events, random_state=42)
        
        # Get unique items from sampled events
        item_col = next((col for col in ['item_id', 'itemid'] if col in events_df.columns), None)
        if not item_col:
            raise ValueError("Could not identify item column in events dataframe")
        
        sampled_items = events_df[item_col].unique()
        
        # Limit number of items if specified
        if max_items and len(sampled_items) > max_items:
            sampled_items = np.random.choice(sampled_items, max_items, replace=False)
            events_df = events_df[events_df[item_col].isin(sampled_items)]
        
        # Filter item properties to only include sampled items
        item_prop_col = next((col for col in ['itemid', 'item_id'] if col in item_properties_df.columns), None)
        if not item_prop_col:
            raise ValueError("Could not identify item column in properties dataframe")
        
        sampled_item_properties_df = item_properties_df[item_properties_df[item_prop_col].isin(sampled_items)]
        
        # Get categories for sampled items
        category_rows = sampled_item_properties_df[
            (sampled_item_properties_df['property'] == 'categoryid') | 
            (sampled_item_properties_df['property'] == 'category_id')
        ]
        sampled_categories = category_rows['value'].unique()
        
        # Identify category and parent columns
        cat_col = next((col for col in ['categoryid', 'category_id'] if col in category_df.columns), None)
        parent_col = next((col for col in ['parentid', 'parent_id'] if col in category_df.columns), None)
        
        if not cat_col:
            raise ValueError(f"Could not identify category column in category dataframe. Available columns: {category_df.columns.tolist()}")
        
        # Limit number of categories if specified
        if max_categories and len(sampled_categories) > max_categories:
            # If we have hierarchy info, try to preserve it by sampling parents first
            if parent_col:
                # Create a mapping of categories to their parents
                cat_to_parent = dict(zip(category_df[cat_col], category_df[parent_col]))
                
                # Identify root categories (those with no parent or NaN parent)
                root_cats = [cat for cat in sampled_categories 
                            if cat not in cat_to_parent or pd.isna(cat_to_parent.get(cat))]
                
                # If we have more root categories than allowed, sample from them
                if len(root_cats) > max_categories // 2:
                    sampled_roots = np.random.choice(root_cats, max_categories // 2, replace=False)
                else:
                    sampled_roots = root_cats
                
                # Start with root categories
                final_categories = set(sampled_roots)
                
                # Add children of sampled roots until we reach max_categories
                remaining_slots = max_categories - len(final_categories)
                if remaining_slots > 0:
                    # Find all children of sampled roots
                    children = []
                    for cat in sampled_categories:
                        if cat_to_parent.get(cat) in final_categories:
                            children.append(cat)
                    
                    # Sample from children if needed
                    if len(children) > remaining_slots:
                        sampled_children = np.random.choice(children, remaining_slots, replace=False)
                        final_categories.update(sampled_children)
                    else:
                        final_categories.update(children)
                
                # If we still have room, add other categories
                remaining_slots = max_categories - len(final_categories)
                if remaining_slots > 0:
                    other_cats = [cat for cat in sampled_categories if cat not in final_categories]
                    if other_cats:
                        if len(other_cats) > remaining_slots:
                            sampled_others = np.random.choice(other_cats, remaining_slots, replace=False)
                            final_categories.update(sampled_others)
                        else:
                            final_categories.update(other_cats)
                
                # Use the final set of categories
                sampled_categories = list(final_categories)
            else:
                # No hierarchy info, just sample randomly
                sampled_categories = np.random.choice(sampled_categories, max_categories, replace=False)
            
            # Filter events again to only include items in sampled categories
            items_in_categories = category_rows[category_rows['value'].isin(sampled_categories)][item_prop_col].unique()
            events_df = events_df[events_df[item_col].isin(items_in_categories)]
            sampled_item_properties_df = sampled_item_properties_df[
                sampled_item_properties_df[item_prop_col].isin(items_in_categories)
            ]
        
        # Make sure we include all parent categories
        relevant_categories = set(sampled_categories)
        
        if parent_col:
            # Get all categories and their parents
            category_hierarchy = dict(zip(category_df[cat_col], category_df[parent_col]))
            
            # Add all parents to relevant categories
            for category in sampled_categories:
                parent = category_hierarchy.get(category)
                while parent and not pd.isna(parent):
                    relevant_categories.add(parent)
                    parent = category_hierarchy.get(parent)
        
        # Filter category dataframe to only include relevant categories and their parents
        sampled_category_df = category_df[category_df[cat_col].isin(relevant_categories)]
        
        # Print statistics
        print(f"Sampled dataset: {len(events_df)} events, {len(sampled_items)} items, "
              f"{len(relevant_categories)} categories")
        
        return events_df, sampled_category_df, sampled_item_properties_df

    def evaluate(self, validation_sessions, k=10, metrics=['hit_rate', 'ndcg', 'mrr']):
        """
        Evaluate the model on validation sessions.
        
        Args:
            validation_sessions: Dictionary of session_id -> session_df
            k: Number of recommendations to generate
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric name -> score
        """
        print(f"Evaluating model on {len(validation_sessions)} validation sessions...")
        
        self.eval()
        results = {
            'hit_rate': 0,
            'ndcg': 0, 
            'mrr': 0,
            'recall': 0,
            'precision': 0
        }
        
        valid_sessions = 0
        
        for session_id, session_df in tqdm(validation_sessions.items()):
            # Get all items in the session
            session_items = session_df['item_id'].tolist()
            
            if len(session_items) < 3:  # Skip very short sessions
                continue
            
            # Use all but last item as input, last item as ground truth
            input_items = session_items[:-1]
            target_item = session_items[-1]
            
            # Get recommendations
            recs = self.predict_next_items(input_items, k=k)
            
            if not recs:  # Skip if no recommendations
                continue
            
            valid_sessions += 1
            
            # Calculate metrics
            if 'hit_rate' in metrics:
                # Hit rate: 1 if target in recommendations, 0 otherwise
                results['hit_rate'] += 1 if target_item in recs else 0
                
            if 'ndcg' in metrics:
                # Normalized Discounted Cumulative Gain
                if target_item in recs:
                    rank = recs.index(target_item) + 1
                    results['ndcg'] += 1 / np.log2(rank + 1)
                    
            if 'mrr' in metrics:
                # Mean Reciprocal Rank
                if target_item in recs:
                    rank = recs.index(target_item) + 1
                    results['mrr'] += 1 / rank
                    
            if 'recall' in metrics:
                # For single target item, recall is same as hit rate
                results['recall'] += 1 if target_item in recs else 0
                
            if 'precision' in metrics:
                # Precision at k
                results['precision'] += (1 if target_item in recs else 0) / min(k, len(recs))
        
        # Normalize results
        if valid_sessions > 0:
            for metric in results:
                results[metric] /= valid_sessions
        
        print(f"Evaluation results (k={k}):")
        for metric, score in results.items():
            if metric in metrics:
                print(f"  {metric}: {score:.4f}")
        
        return results

    def fit_with_validation(self, events_df, category_df, item_properties_df, batch_size=1024, 
                            n_epochs=5, validation_ratio=0.2, early_stopping=True, patience=2,
                            max_events=None, max_items=None, max_categories=None):
        """
        Fit the model with validation.
        
        Args:
            events_df: DataFrame with user events
            category_df: DataFrame with category hierarchy
            item_properties_df: DataFrame with item properties
            batch_size: Batch size for training
            n_epochs: Maximum number of training epochs
            validation_ratio: Ratio of sessions to use for validation
            early_stopping: Whether to use early stopping
            patience: Number of epochs with no improvement before stopping
            max_events: Maximum number of events to use
            max_items: Maximum number of unique items
            max_categories: Maximum number of categories
            
        Returns:
            self
        """
        print("Building hierarchical Markov model with validation...")
        
        # Sample dataset if limits are specified
        if max_events or max_items or max_categories:
            events_df, category_df, item_properties_df = self.sample_dataset(
                events_df, category_df, item_properties_df,
                max_events=max_events, max_items=max_items, max_categories=max_categories
            )
        
        # Process category hierarchy
        self._process_category_hierarchy(category_df)
        
        # Process item properties
        self._process_item_properties(item_properties_df)
        
        # Process events and create sessions
        all_sessions = self._identify_sessions(events_df)
        
        # Split sessions for training and validation
        session_ids = list(all_sessions.keys())
        np.random.shuffle(session_ids)
        
        val_size = int(len(session_ids) * validation_ratio)
        train_ids = session_ids[val_size:]
        val_ids = session_ids[:val_size]
        
        train_sessions = {sid: all_sessions[sid] for sid in train_ids}
        val_sessions = {sid: all_sessions[sid] for sid in val_ids}
        
        print(f"Split into {len(train_sessions)} training and {len(val_sessions)} validation sessions")
        
        # Build transition tensors from training sessions
        self._build_transition_tensors(train_sessions)
        
        # Initialize embedding layers
        self._init_embedding_layers()
        
        # Training loop with validation
        self.train()
        best_score = 0
        best_epoch = 0
        no_improvement = 0
        
        # Dataset for training
        train_data = []
        for session_id, session_df in train_sessions.items():
            items = session_df['item_id'].tolist()
            if len(items) >= 3:  # Need at least 3 items for training
                valid_items = [
                    item for item in items 
                    if item in self.item_encoder.classes_ and item in self.item_to_category
                ]
                if len(valid_items) >= 3:
                    encoded_items = self.item_encoder.transform(valid_items).tolist()
                    train_data.append(encoded_items)
        
        # Create data loader
        dataset = SessionDataset(train_data)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=session_collate_fn
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        for epoch in range(n_epochs):
            # Train for one epoch
            total_loss = self._train_epoch(data_loader, optimizer)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(val_sessions, k=10, metrics=['hit_rate', 'ndcg'])
            val_score = val_metrics['hit_rate']  # Use hit rate as the main metric
            
            print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {total_loss:.4f}, Validation Score: {val_score:.4f}")
            
            # Check for improvement
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                no_improvement = 0
                # Save the best model (could add actual saving here)
                best_model_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                no_improvement += 1
            
            # Early stopping
            if early_stopping and no_improvement >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # If using early stopping, revert to best model
        if early_stopping and best_epoch < epoch:
            print(f"Reverting to best model from epoch {best_epoch+1}")
            self.load_state_dict(best_model_state)
        
        print("Hierarchical Markov model trained successfully")
        return self

    def _train_epoch(self, data_loader, optimizer):
        """Train for a single epoch."""
        self.train()
        total_loss = 0
        
        for batch in tqdm(data_loader, desc="Training"):
            # Get context and target
            context_items, target_items = batch
            
            # Convert to device
            context_items = [torch.tensor(session, dtype=torch.long, device=self.device) 
                           for session in context_items]
            target_items = torch.tensor(target_items, dtype=torch.long, device=self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            batch_loss = 0
            for i, (context, target) in enumerate(zip(context_items, target_items)):
                # Check for empty context
                if len(context) == 0:
                    continue
                    
                # Get item embeddings for context
                item_embeds = self.item_embeddings(context)
                
                # Encode context safely
                context_packed = nn.utils.rnn.pack_sequence([item_embeds])
                _, (hidden, _) = self.context_encoder(context_packed)
                # Safe squeezing
                context_encoding = hidden[0] if hidden.size(0) == 1 else hidden.squeeze(0)
                
                # Get target item embedding
                target_embed = self.item_embeddings(target)
                
                # Get target category with error handling
                target_category = self.item_category_tensor[target]
                if target_category == -1:
                    # Skip items with unknown category
                    continue
                
                # Category prediction loss
                cat_pred_input = torch.cat([context_encoding, target_embed], dim=0).unsqueeze(0)
                cat_pred = self.category_predictor(cat_pred_input)
                cat_loss = F.cross_entropy(cat_pred, target_category.unsqueeze(0))
                
                # Item prediction loss
                item_pred_input = torch.cat([
                    context_encoding, 
                    target_embed, 
                    self.category_embeddings(target_category)
                ], dim=0).unsqueeze(0)
                
                item_pred = self.item_predictor(item_pred_input)
                item_loss = F.binary_cross_entropy(item_pred, torch.ones(1, 1, device=self.device))
                
                # Combined loss
                batch_loss += cat_loss + item_loss
            
            # Check if we had any valid examples
            if batch_loss > 0:
                # Average loss for the batch
                batch_loss /= len(target_items)
                
                # Backward pass
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
        
        # Average loss over all batches
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        return avg_loss


class SessionDataset(Dataset):
    """Dataset for training on session data."""
    
    def __init__(self, sessions, max_len=10):
        """
        Initialize dataset.
        
        Args:
            sessions: List of sessions, where each session is a list of encoded items
            max_len: Maximum context length to use
        """
        self.sessions = sessions
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = self.sessions[idx]
        
        # We'll use all but the last item as context, and the last item as target
        if len(session) <= self.max_len:
            context = session[:-1]
            target = session[-1]
        else:
            # Use the last max_len+1 items
            context = session[-self.max_len-1:-1]
            target = session[-1]
        
        return context, target


def session_collate_fn(batch):
    """Collate function for SessionDataset."""
    contexts = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return contexts, targets


# Example usage:
if __name__ == "__main__":
    # Load data
    events_df = pd.read_csv("data/events.csv")
    category_df = pd.read_csv("data/category_tree.csv")
    item_props_df = pd.concat([
        pd.read_csv("data/item_properties_part1.csv"),
        pd.read_csv("data/item_properties_part2.csv")
    ])
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HierarchicalMarkovModel(embedding_dim=64, device=device)
    
    # Sample the dataset first
    max_events = 50000  # Increased for better validation
    max_items = 10000
    sampled_events, sampled_categories, sampled_props = model.sample_dataset(
        events_df, category_df, item_props_df,
        max_events=max_events, max_items=max_items
    )
    
    # Fit model with validation
    model.fit_with_validation(
        sampled_events, sampled_categories, sampled_props, 
        batch_size=128, n_epochs=5, validation_ratio=0.2,
        early_stopping=True, patience=2
    )
    
    # Test on a sample session
    test_df = sampled_events.copy()
    if 'datetime' not in test_df.columns and 'timestamp' in test_df.columns:
        test_df['datetime'] = pd.to_datetime(test_df['timestamp'])
    if 'item_id' not in test_df.columns and 'itemid' in test_df.columns:
        test_df['item_id'] = test_df['itemid']
    
    # Find the longest session for testing
    sessions = model._identify_sessions(test_df)
    longest_session = None
    max_length = 0
    
    for sid, session in sessions.items():
        if len(session) > max_length:
            max_length = len(session)
            longest_session = session
    
    if longest_session is not None:
        test_items = longest_session['item_id'].tolist()
        # Use all but last 5 items to predict the next 5
        input_items = test_items[:-5]
        target_items = test_items[-5:]
        
        print(f"Testing with {len(input_items)} input items to predict the next 5 items")
        print(f"Ground truth: {target_items}")
        
        recommendations = model.predict_next_items(input_items, k=10)
        print(f"Recommendations: {recommendations}")
        
        # Calculate matches
        matches = set(recommendations).intersection(set(target_items))
        print(f"Matched {len(matches)} out of 5 items: {matches}")