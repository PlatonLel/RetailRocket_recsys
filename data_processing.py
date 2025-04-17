import pandas as pd
import numpy as np
import os
from collections import defaultdict
import scipy.sparse as sp
from sklearn import preprocessing

INTERACTION_WEIGHTS = {'view': 1.0, 'addtocart': 2.0, 'transaction': 3.0}

def process_retailrocket_data(data_path, event_limit=None, weights=INTERACTION_WEIGHTS, min_interactions=2):
    print("Loading events...")
    events = pd.read_csv(os.path.join(data_path, 'events.csv'))
    if event_limit:
        events = events[:event_limit]

    print("Loading item properties...")
    items_part1 = pd.read_csv(os.path.join(data_path, 'item_properties_part1.csv'))
    items_part2 = pd.read_csv(os.path.join(data_path, 'item_properties_part2.csv'))
    items = pd.concat([items_part1, items_part2], ignore_index=True)
    del items_part1, items_part2
    if event_limit:
        items = items[:event_limit]

    
    events = events.rename(columns={'visitorid': 'orig_user_id', 'itemid': 'orig_item_id',
                                          'timestamp': 'event_time', 'event': 'event_type'})
    events['event_time'] = pd.to_datetime(events['event_time'], unit='ms')
    events = events.dropna(subset=['orig_user_id', 'orig_item_id'])
    events['orig_user_id'] = events['orig_user_id'].astype(int)
    events['orig_item_id'] = events['orig_item_id'].astype(int)

    items = items.rename(columns={'timestamp': 'prop_time', 'property': 'prop_name', 'value': 'prop_value', 'itemid': 'orig_item_id'})
    items['prop_time'] = pd.to_datetime(items['prop_time'], unit='ms')
    items = items.dropna(subset=['orig_item_id', 'prop_name', 'prop_value'])
    items['orig_item_id'] = items['orig_item_id'].astype(int)

    print("Merging events with time-correct properties...")
    merged = pd.merge_asof(
        events.sort_values('event_time'),
        items[['orig_item_id', 'prop_time', 'prop_name', 'prop_value']].sort_values('prop_time'),
        left_on='event_time',
        right_on='prop_time',
        by='orig_item_id',
        direction='backward'
    )
    merged = merged.dropna(subset=['prop_time'])

    print("Mapping IDs and features...")
    user_encoder = preprocessing.LabelEncoder()
    item_encoder = preprocessing.LabelEncoder()
    prop_name_encoder = preprocessing.LabelEncoder()
    prop_value_encoder = preprocessing.LabelEncoder()

    merged['user_idx'] = user_encoder.fit_transform(merged['orig_user_id'])
    merged['item_idx'] = item_encoder.fit_transform(merged['orig_item_id'])
    merged['prop_type_idx'] = prop_name_encoder.fit_transform(merged['prop_name'])
    merged['prop_val_idx'] = prop_value_encoder.fit_transform(merged['prop_value'].astype(str))
    merged['weight'] = merged['event_type'].map(weights).fillna(0.0)

    merged = merged.dropna(subset=['user_idx', 'item_idx', 'prop_type_idx', 'prop_val_idx', 'weight'])
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    prop_type_vocab_size = len(prop_name_encoder.classes_)
    prop_value_vocab_size = len(prop_value_encoder.classes_)

    print("Extracting latest item features...")
    latest_features_df = merged[['item_idx', 'prop_type_idx', 'prop_val_idx', 'event_time']].copy()
    latest_features_df = latest_features_df.sort_values('event_time', ascending=False)
    latest_features_df = latest_features_df.drop_duplicates(subset=['item_idx', 'prop_type_idx'], keep='first')
    
    latest_item_features_map = defaultdict(dict)
    for _, row in latest_features_df.iterrows():
         latest_item_features_map[int(row['item_idx'])][int(row['prop_type_idx'])] = int(row['prop_val_idx'])

    print("Splitting train/test data (leave-one-out)...")
    merged = merged.sort_values(['user_idx', 'event_time'])
    merged['is_last'] = (merged.groupby('user_idx').cumcount(ascending=False) == 0)
    
    user_interaction_counts = merged.groupby('user_idx').size()
    users_with_one_interaction = user_interaction_counts[user_interaction_counts >= min_interactions].index
    
    filtered_merged = merged[merged['user_idx'].isin(users_with_one_interaction)].copy()
    del merged
    print(f"Filtered merged length: {len(filtered_merged)}")
    latest_features = filtered_merged[['item_idx', 'prop_type_idx', 'prop_val_idx', 'event_time']].copy()
    latest_features = latest_features.sort_values('event_time', ascending=False)
    latest_features = latest_features.drop_duplicates(subset=['item_idx', 'prop_type_idx'], keep='first')
    latest_item_features_map = defaultdict(dict)
    for _, row in latest_features.iterrows():
        latest_item_features_map[int(row['item_idx'])][int(row['prop_type_idx'])] = int(row['prop_val_idx'])

    filtered_merged = filtered_merged.sort_values(['user_idx', 'event_time'])
    filtered_merged['is_last'] = (filtered_merged.groupby('user_idx').cumcount(ascending=False) == 0)

    test_mask = filtered_merged['is_last']
    train_mask = ~test_mask
    

    relevant_cols = ['user_idx', 'item_idx', 'weight', 'prop_type_idx', 'prop_val_idx']
    train_data = filtered_merged.loc[train_mask, relevant_cols].values.astype(np.float32)
    test_data = filtered_merged.loc[test_mask, relevant_cols].values.astype(np.float32)

    print("Building user history for CBF...")
    user_history_for_cbf = defaultdict(list)
    for row in train_data:
        user_idx, item_idx, _, prop_type_idx, prop_val_idx = row
        user_history_for_cbf[int(user_idx)].append((int(item_idx), int(prop_type_idx), int(prop_val_idx)))
    train_data = train_data[:, :3]
    test_data = test_data[:, :3]
    print("Data processing complete.")
    return {
        "train_data": train_data,
        "test_data": test_data,
        "num_users": num_users,
        "num_items": num_items,
        "user_history_for_cbf": user_history_for_cbf,
        "latest_item_features_map": dict(latest_item_features_map),
        "prop_type_vocab_size": prop_type_vocab_size,
        "prop_value_vocab_size": prop_value_vocab_size,
        "user_encoder": user_encoder,
        "item_encoder": item_encoder,
        "prop_name_encoder": prop_name_encoder,
        "prop_value_encoder": prop_value_encoder,
    }

def build_item_feature_matrix(latest_item_features_map, num_items, prop_type_vocab_size, prop_value_vocab_size):
    num_feature_dimensions = prop_type_vocab_size + prop_value_vocab_size
    rows, cols, vals = [], [], []
    
    for item_idx in range(num_items):
        features = latest_item_features_map.get(item_idx, {})
        for prop_type_idx, prop_val_idx in features.items():
            if 0 <= prop_type_idx < prop_type_vocab_size:
                rows.append(item_idx)
                cols.append(prop_type_idx)
                vals.append(1)
            
            if 0 <= prop_val_idx < prop_value_vocab_size:
                col_idx = prop_type_vocab_size + prop_val_idx
                rows.append(item_idx)
                cols.append(col_idx)
                vals.append(1)

    X = sp.coo_matrix((vals, (rows, cols)), shape=(num_items, num_feature_dimensions)).tocsr()
    X.sum_duplicates()
    X.data[:] = 1
    print(f"Built item feature matrix with shape: {X.shape}")
    return X


