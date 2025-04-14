import pandas as pd
import numpy as np
import os
from collections import defaultdict
import random

def data_processing(data_path, session_length=30):
    print("Loading data...")
    events = pd.read_csv(os.path.join(data_path, 'events.csv'))
    items_part1 = pd.read_csv(os.path.join(data_path, 'item_properties_part1.csv'))
    items_part2 = pd.read_csv(os.path.join(data_path, 'item_properties_part2.csv'))
    items = pd.concat([items_part1, items_part2], ignore_index=True)
    del items_part1, items_part2

    # Category mapping
    items.loc[items['property'] == 'categoryid', 'property'] = 1698
    items.loc[items['property'] == 'available', 'property'] = 1697
    item_cat_prop = items[items['property'] == 1698][['itemid', 'value']].copy()
    item_cat_prop = item_cat_prop.rename(columns={'value': 'category_id'})
    item_cat_prop['category_id'] = pd.to_numeric(item_cat_prop['category_id'], errors='coerce')
    item_cat_prop.dropna(subset=['category_id'], inplace=True)
    item_cat_prop['category_id'] = item_cat_prop['category_id'].astype(int)
    item_to_category_map = item_cat_prop.groupby('itemid')['category_id'].first().to_dict()

    events = events.rename(columns={'visitorid': 'visitor_id', 'event': 'event_type', 'timestamp': 'event_time'})
    events['event_time'] = pd.to_datetime(events['event_time'], unit='ms')
    if 'transactionid' in events.columns:
        events.drop(columns=['transactionid'], inplace=True)
    event_core_cols = ['visitor_id', 'itemid', 'event_time', 'event_type']
    core_events = events[event_core_cols].drop_duplicates().copy()
    core_events = core_events.sort_values(['visitor_id', 'event_time'])

    items = items.rename(columns={'timestamp': 'property_time', 'value': 'property_value'})
    items['property_time'] = pd.to_datetime(items['property_time'], unit='ms')
    items = items.sort_values(['itemid', 'property_time'])

    events_with_properties = pd.merge_asof(
        core_events.sort_values('event_time'),
        items[['itemid', 'property_time', 'property', 'property_value']].sort_values('property_time'),
        left_on='event_time',
        right_on='property_time',
        by='itemid',
        direction='backward'
    )
    events_with_properties.dropna(subset=['property_time'], inplace=True)
    events_with_properties = events_with_properties.sort_values(['visitor_id', 'event_time', 'property_time'])
    final_events = events_with_properties.drop_duplicates(subset=event_core_cols, keep='first').copy()
    final_events['time_diff'] = final_events.groupby('visitor_id')['event_time'].diff().dt.total_seconds()
    final_events['session_id'] = ((final_events['time_diff'] > session_length*60) | (final_events['time_diff'].isna())).astype(int).groupby(final_events['visitor_id']).cumsum()

    session_grouped_events = final_events.sort_values('event_time').groupby(['visitor_id', 'session_id'])

    item_sequences = []
    category_sequences = []
    property_type_sequences_orig = []
    property_value_sequences_str = []
    for (visitor_id, session_id), session_df in session_grouped_events:
        items_in_session = session_df['itemid'].tolist()
        cats_in_session = [item_to_category_map.get(i, 0) for i in items_in_session]
        if len(items_in_session) > 1:
            item_sequences.append(items_in_session)
            category_sequences.append(cats_in_session)
            property_type_sequences_orig.append(session_df['property'].fillna(-1).astype(int).tolist())
            property_value_sequences_str.append(session_df['property_value'].fillna('PAD').astype(str).tolist())

    all_prop_values_flat = [val for seq in property_value_sequences_str for val in seq]
    unique_prop_values = sorted(list(set(all_prop_values_flat)))
    prop_value_map = {val: i + 1 for i, val in enumerate(unique_prop_values) if val != 'PAD'}
    prop_value_map['PAD'] = 0
    property_value_sequences_int = []
    for seq_str in property_value_sequences_str:
        property_value_sequences_int.append([prop_value_map.get(val, 0) for val in seq_str])

    all_prop_types_flat = [ptype for seq in property_type_sequences_orig for ptype in seq]
    unique_prop_types_orig = sorted([p for p in list(set(all_prop_types_flat)) if p != -1])
    prop_type_map = {orig_code: i + 1 for i, orig_code in enumerate(unique_prop_types_orig)}
    prop_type_map[-1] = 0
    property_type_sequences_mapped = []
    for seq_orig in property_type_sequences_orig:
        property_type_sequences_mapped.append([prop_type_map.get(orig_code, 0) for orig_code in seq_orig])

    all_items = final_events['itemid'].unique().astype(int)
    all_categories = item_cat_prop['category_id'].unique().astype(int)
    prop_type_vocab_size = len(prop_type_map)
    prop_value_vocab_size = len(prop_value_map)
    item_vocab_size = int(np.max(all_items)) + 1 if len(all_items) > 0 else 1
    category_vocab_size = int(np.max(all_categories)) + 1 if len(all_categories) > 0 else 1
    user_vocab_size = len(item_sequences)
    print(item_vocab_size, category_vocab_size, prop_type_vocab_size, prop_value_vocab_size, user_vocab_size)
    return {
        'item_sequences': item_sequences,
        'category_sequences': category_sequences,
        'property_type_sequences': property_type_sequences_mapped,
        'property_value_sequences': property_value_sequences_int,
        'item_vocab_size': item_vocab_size,
        'category_vocab_size': category_vocab_size,
        'prop_type_vocab_size': prop_type_vocab_size,
        'prop_value_vocab_size': prop_value_vocab_size,
        'user_vocab_size': user_vocab_size,
    }

def prepare_bpr_data(item_sequences, category_sequences, property_type_sequences, property_value_sequences):
    print("Preparing BPR data...")
    user_item_dict = {}
    all_items = set()
    item_to_cat = {}
    item_to_prop_type = {}
    item_to_prop_value = {}
    for seq, cats, prop_types, prop_values in zip(item_sequences, category_sequences, property_type_sequences, property_value_sequences):
        for item, cat, ptype, pval in zip(seq, cats, prop_types, prop_values):
            if item not in item_to_cat:
                item_to_cat[item] = cat
            if item not in item_to_prop_type:
                item_to_prop_type[item] = ptype
            if item not in item_to_prop_value:
                item_to_prop_value[item] = pval
    for user_id, seq in enumerate(item_sequences):
        user_item_dict[user_id] = set(seq)
        all_items.update(seq)
    all_items = list(all_items)
    bpr_triplets = []
    for user_id, pos_items in user_item_dict.items():
        for pos_item in pos_items:
            neg_item = random.choice(all_items)
            while neg_item in pos_items:
                neg_item = random.choice(all_items)
            pos_cat = item_to_cat.get(pos_item, 0)
            neg_cat = item_to_cat.get(neg_item, 0)
            pos_prop_type = item_to_prop_type.get(pos_item, 0)
            neg_prop_type = item_to_prop_type.get(neg_item, 0)
            pos_prop_value = item_to_prop_value.get(pos_item, 0)
            neg_prop_value = item_to_prop_value.get(neg_item, 0)
            bpr_triplets.append((user_id, pos_item, neg_item, pos_cat, neg_cat, pos_prop_type, neg_prop_type, pos_prop_value, neg_prop_value))
    return bpr_triplets, user_item_dict, all_items