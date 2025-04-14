import pandas as pd
import numpy as np
import os

def data_processing(data_path, session_length=30):
    print("Loading data...")
    events = pd.read_csv(os.path.join(data_path, 'events.csv'))
    categories = pd.read_csv(os.path.join(data_path, 'category_tree.csv'))
    items_part1 = pd.read_csv(os.path.join(data_path, 'item_properties_part1.csv'))
    items_part2 = pd.read_csv(os.path.join(data_path, 'item_properties_part2.csv'))
    items = pd.concat([items_part1, items_part2], ignore_index=True)
    del items_part1, items_part2

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

    print("Preparing core events...")
    events = events.rename(columns={
        'visitorid': 'visitor_id',
        'event': 'event_type',
        'timestamp': 'event_time'
    })
    events['event_time'] = pd.to_datetime(events['event_time'], unit='ms')
    if 'transactionid' in events.columns:
        events.drop(columns=['transactionid'], inplace=True)

    event_core_cols = ['visitor_id', 'itemid', 'event_time', 'event_type']
    core_events = events[event_core_cols].drop_duplicates().copy()
    core_events = core_events.sort_values(['visitor_id', 'event_time']) # Essential for session logic

    print("Preparing item properties...")
    items = items.rename(columns={'timestamp': 'property_time', 'value': 'property_value'})
    items['property_time'] = pd.to_datetime(items['property_time'], unit='ms')
    items = items.sort_values(['itemid', 'property_time'])

    print("Merging properties with events temporally...")
    events_with_properties = pd.merge_asof(
        core_events.sort_values('event_time'),
        items[['itemid', 'property_time', 'property', 'property_value']].sort_values('property_time'), # Select only needed columns
        left_on='event_time',
        right_on='property_time',
        by='itemid',
        direction='backward'
    )
    events_with_properties.dropna(subset=['property_time'], inplace=True)

    print("Selecting first property match per event...")
    events_with_properties = events_with_properties.sort_values(['visitor_id', 'event_time', 'property_time'])
    final_events = events_with_properties.drop_duplicates(subset=event_core_cols, keep='first').copy()

    print("Performing sessionization...")
    final_events['time_diff'] = final_events.groupby('visitor_id')['event_time'].diff().dt.total_seconds()
    final_events['session_id'] = ((final_events['time_diff'] > session_length*60) | (final_events['time_diff'].isna())).astype(int).groupby(final_events['visitor_id']).cumsum()

    print("Adding category and weights...")
    final_events['category_id'] = final_events['itemid'].map(item_to_category_map)
    final_events['category_id'] = final_events['category_id'].fillna(0).astype(int)

    print("Preparing sequences...")
    session_grouped_events = final_events.sort_values('event_time').groupby(['visitor_id', 'session_id'])

    item_sequences = []
    category_sequences = []

    property_type_sequences_orig = []
    property_value_sequences_str = []

    for (visitor_id, session_id), session_df in session_grouped_events:
        items_in_session = session_df['itemid'].tolist()
        if len(items_in_session) > 1:
            item_sequences.append(items_in_session)
            category_sequences.append(session_df['category_id'].tolist())
            property_type_sequences_orig.append(session_df['property'].fillna(-1).astype(int).tolist())
            property_value_sequences_str.append(session_df['property_value'].fillna('PAD').astype(str).tolist())

    print("Building property value vocabulary...")
    all_prop_values_flat = [val for seq in property_value_sequences_str for val in seq]
    unique_prop_values = sorted(list(set(all_prop_values_flat)))
    prop_value_map = {val: i + 1 for i, val in enumerate(unique_prop_values) if val != 'PAD'}
    prop_value_map['PAD'] = 0
    property_value_sequences_int = []
    for seq_str in property_value_sequences_str:
        property_value_sequences_int.append([prop_value_map.get(val, 0) for val in seq_str])


    print("Building property type vocabulary...")
    all_prop_types_flat = [ptype for seq in property_type_sequences_orig for ptype in seq]
    unique_prop_types_orig = sorted([p for p in list(set(all_prop_types_flat)) if p != -1])

    prop_type_map = {orig_code: i + 1 for i, orig_code in enumerate(unique_prop_types_orig)}
    prop_type_map[-1] = 0

    property_type_sequences_mapped = []
    for seq_orig in property_type_sequences_orig:
        property_type_sequences_mapped.append([prop_type_map.get(orig_code, 0) for orig_code in seq_orig])

    print("Calculating vocab sizes...")
    all_items = final_events['itemid'].unique().astype(int)
    all_categories = final_events['category_id'].unique().astype(int)
    prop_type_vocab_size = len(prop_type_map)
    prop_value_vocab_size = len(prop_value_map)

    item_vocab_size = int(np.max(all_items)) + 1 if len(all_items) > 0 else 1
    category_vocab_size = int(np.max(all_categories)) + 1 if len(all_categories) > 0 else 1


    print("Data processing finished.")
    print(f"Property Value Vocab size: {prop_value_vocab_size}")
    print(f"Property Type Vocab size (Mapped): {prop_type_vocab_size}")
    print(f"Item vocab size: {item_vocab_size}")
    print(f"Category vocab size: {category_vocab_size}")
    return {
        'item_sequences': item_sequences,
        'category_sequences': category_sequences,
        'property_type_sequences': property_type_sequences_mapped,
        'property_value_sequences': property_value_sequences_int,
        'item_vocab_size': item_vocab_size,
        'category_vocab_size': category_vocab_size,
        'prop_type_vocab_size': prop_type_vocab_size,
        'prop_value_vocab_size': prop_value_vocab_size,
    }