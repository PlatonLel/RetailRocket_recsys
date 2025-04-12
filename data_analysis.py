# data_analysis.py
"""
E-commerce Data Analysis
-----------------------
Comprehensive analysis of available e-commerce datasets for recommendation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "C:/Users/lelik/mine/e-commerce-recommender/data"

def analyze_events_data():
    """Analyze user event data."""
    print("\n" + "="*50)
    print("EVENTS DATA ANALYSIS")
    print("="*50)
    
    events_path = Path(DATA_PATH) / "events.csv"
    print(f"Loading events data from {events_path}...")
    
    # Read a sample first to understand structure
    events_sample = pd.read_csv(events_path, nrows=5)
    print("\nEvents data sample:")
    print(events_sample)
    print(f"\nColumns: {events_sample.columns.tolist()}")
    
    # Read in chunks to handle large file
    chunk_size = 100000
    chunks = pd.read_csv(events_path, chunksize=chunk_size)
    
    # Initialize counters
    total_events = 0
    user_count = set()
    item_count = set()
    event_types = {}
    unique_days = set()
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}... ({chunk_size} rows)")
        
        # Update counters
        total_events += len(chunk)
        user_count.update(chunk['visitorid'].unique())
        item_count.update(chunk['itemid'].unique())
        
        # Count event types
        for event, count in chunk['event'].value_counts().items():
            event_types[event] = event_types.get(event, 0) + count
        
        # Extract dates
        dates = pd.to_datetime(chunk['timestamp'], unit='ms').dt.date
        unique_days.update(dates.unique())
        
        # Sample for time analysis
        if i == 0:
            time_sample = chunk
    
    # Time analysis
    time_sample['datetime'] = pd.to_datetime(time_sample['timestamp'], unit='ms')
    time_sample['date'] = time_sample['datetime'].dt.date
    time_sample['hour'] = time_sample['datetime'].dt.hour
    
    # Print results
    print("\nEvents Summary Statistics:")
    print(f"Total events: {total_events:,}")
    print(f"Unique users: {len(user_count):,}")
    print(f"Unique items: {len(item_count):,}")
    print(f"Event types distribution: {event_types}")
    print(f"Date range: {min(unique_days)} to {max(unique_days)}")
    print(f"Total days: {len(unique_days)}")
    
    # Analyze user activity
    print("\nAnalyzing user activity...")
    events_per_user = {}
    for chunk in pd.read_csv(events_path, chunksize=chunk_size):
        user_counts = chunk['visitorid'].value_counts().to_dict()
        for user, count in user_counts.items():
            events_per_user[user] = events_per_user.get(user, 0) + count
    
    events_per_user_series = pd.Series(events_per_user)
    print(f"Events per user statistics:")
    print(f"  Min: {events_per_user_series.min()}")
    print(f"  25%: {events_per_user_series.quantile(0.25)}")
    print(f"  Median: {events_per_user_series.median()}")
    print(f"  75%: {events_per_user_series.quantile(0.75)}")
    print(f"  Max: {events_per_user_series.max()}")
    print(f"  Mean: {events_per_user_series.mean():.2f}")
    
    # Clean up
    del events_per_user, events_per_user_series
    gc.collect()
    
    return {
        'total_events': total_events,
        'user_count': len(user_count),
        'item_count': len(item_count),
        'event_types': event_types,
        'date_range': (min(unique_days), max(unique_days))
    }

def analyze_item_properties():
    """Analyze item properties data."""
    print("\n" + "="*50)
    print("ITEM PROPERTIES ANALYSIS")
    print("="*50)
    
    properties_path1 = Path(DATA_PATH) / "item_properties_part1.csv"
    properties_path2 = Path(DATA_PATH) / "item_properties_part2.csv"
    
    print(f"Loading item properties from {properties_path1} and {properties_path2}...")
    
    # Read samples to understand structure
    props_sample1 = pd.read_csv(properties_path1, nrows=5)
    print("\nItem properties sample (part 1):")
    print(props_sample1)
    print(f"\nColumns: {props_sample1.columns.tolist()}")
    
    # Initialize counters
    total_properties = 0
    item_count = set()
    property_types = set()
    
    # Process part 1
    chunk_size = 100000
    for i, chunk in enumerate(pd.read_csv(properties_path1, chunksize=chunk_size)):
        print(f"Processing part 1, chunk {i+1}... ({chunk_size} rows)")
        total_properties += len(chunk)
        item_count.update(chunk['itemid'].unique())
        property_types.update(chunk['property'].unique())
    
    # Process part 2
    for i, chunk in enumerate(pd.read_csv(properties_path2, chunksize=chunk_size)):
        print(f"Processing part 2, chunk {i+1}... ({chunk_size} rows)")
        total_properties += len(chunk)
        item_count.update(chunk['itemid'].unique())
        property_types.update(chunk['property'].unique())
    
    # Analyze property types - load a larger sample
    sample_size = 500000
    props_sample = pd.concat([
        pd.read_csv(properties_path1, nrows=sample_size//2),
        pd.read_csv(properties_path2, nrows=sample_size//2)
    ])
    property_distribution = props_sample['property'].value_counts().to_dict()
    
    # Print results
    print("\nItem Properties Summary Statistics:")
    print(f"Total property records: {total_properties:,}")
    print(f"Unique items: {len(item_count):,}")
    print(f"Unique property types: {len(property_types)}")
    
    print("\nTop 20 property types by frequency:")
    for prop, count in sorted(property_distribution.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {prop}: {count:,} occurrences")
    
    # Analyze specific important properties
    print("\nAnalyzing important properties...")
    
    # Category analysis
    category_items = props_sample[props_sample['property'] == 'categoryid']
    print(f"\nCategory stats:")
    print(f"  Items with category: {category_items['itemid'].nunique():,}")
    print(f"  Unique categories: {category_items['value'].nunique():,}")
    
    # Brand analysis
    brand_items = props_sample[props_sample['property'] == 'brand']
    print(f"\nBrand stats:")
    print(f"  Items with brand: {brand_items['itemid'].nunique():,}")
    print(f"  Unique brands: {brand_items['value'].nunique():,}")
    
    # Price analysis
    price_items = props_sample[props_sample['property'] == 'price']
    if not price_items.empty:
        price_items['value'] = pd.to_numeric(price_items['value'], errors='coerce')
        price_items = price_items.dropna(subset=['value'])
        print(f"\nPrice stats:")
        print(f"  Items with price: {price_items['itemid'].nunique():,}")
        print(f"  Price range: {price_items['value'].min():.2f} - {price_items['value'].max():.2f}")
        print(f"  Median price: {price_items['value'].median():.2f}")
    
    # Clean up
    del props_sample, category_items, brand_items
    if 'price_items' in locals():
        del price_items
    gc.collect()
    
    return {
        'total_properties': total_properties,
        'item_count': len(item_count),
        'property_types': list(property_types)
    }

def analyze_category_tree():
    """Analyze category hierarchy data."""
    print("\n" + "="*50)
    print("CATEGORY TREE ANALYSIS")
    print("="*50)
    
    category_path = Path(DATA_PATH) / "category_tree.csv"
    print(f"Loading category tree from {category_path}...")
    
    # Load the category data (it's small enough to load completely)
    category_data = pd.read_csv(category_path)
    
    print("\nCategory tree sample:")
    print(category_data.head())
    print(f"\nColumns: {category_data.columns.tolist()}")
    
    # Basic statistics
    total_categories = category_data['categoryid'].nunique()
    root_categories = category_data[category_data['parentid'].isna()]['categoryid'].nunique()
    
    # Level distribution
    levels = {}
    
    # Identify root categories
    level_0_categories = set(category_data[category_data['parentid'].isna()]['categoryid'])
    levels[0] = len(level_0_categories)
    
    # Identify categories at each level
    current_level = 0
    current_parents = level_0_categories
    
    while current_parents:
        current_level += 1
        # Find children of current parents
        child_mask = category_data['parentid'].isin(current_parents)
        children = set(category_data[child_mask]['categoryid'])
        
        if not children:
            break
            
        levels[current_level] = len(children)
        current_parents = children
    
    print("\nCategory Tree Summary Statistics:")
    print(f"Total categories: {total_categories:,}")
    print(f"Root categories: {root_categories:,}")
    
    print("\nCategories by level:")
    for level, count in levels.items():
        print(f"  Level {level}: {count:,} categories")
         
    # Calculate average children per parent
    category_data['has_parent'] = ~category_data['parentid'].isna()
    children_per_parent = category_data['has_parent'].sum() / (total_categories - root_categories)
    
    print(f"\nAverage children per parent category: {children_per_parent:.2f}")
    
    return {
        'total_categories': total_categories,
        'root_categories': root_categories,
    }

def analyze_cross_dataset_relationships():
    """Analyze relationships between datasets."""
    print("\n" + "="*50)
    print("CROSS-DATASET RELATIONSHIP ANALYSIS")
    print("="*50)
    
    # Load small samples of each dataset
    events_sample = pd.read_csv(Path(DATA_PATH) / "events.csv", nrows=100000)
    props_sample1 = pd.read_csv(Path(DATA_PATH) / "item_properties_part1.csv", nrows=100000)
    props_sample2 = pd.read_csv(Path(DATA_PATH) / "item_properties_part2.csv", nrows=100000)
    props_sample = pd.concat([props_sample1, props_sample2])
    category_data = pd.read_csv(Path(DATA_PATH) / "category_tree.csv")
    
    # Items in events vs. items in properties
    event_items = set(events_sample['itemid'].unique())
    prop_items = set(props_sample['itemid'].unique())
    
    print("\nItem coverage:")
    print(f"Items in events sample: {len(event_items):,}")
    print(f"Items in properties sample: {len(prop_items):,}")
    print(f"Items in both: {len(event_items & prop_items):,}")
    print(f"Coverage ratio: {len(event_items & prop_items) / len(event_items):.2%}")
    
    # Category coverage
    category_items = props_sample[props_sample['property'] == 'categoryid']
    event_categories = set()
    
    for item in event_items:
        cat_rows = category_items[category_items['itemid'] == item]
        if not cat_rows.empty:
            event_categories.update(cat_rows['value'].unique())
    
    tree_categories = set(category_data['categoryid'].unique())
    
    print("\nCategory coverage:")
    print(f"Categories from items in events: {len(event_categories):,}")
    print(f"Categories in category tree: {len(tree_categories):,}")
    print(f"Categories in both: {len(event_categories & tree_categories):,}")
    if event_categories:
        print(f"Coverage ratio: {len(event_categories & tree_categories) / len(event_categories):.2%}")
    
    # Clean up
    del events_sample, props_sample, props_sample1, props_sample2, category_items
    gc.collect()

def identify_potential_features():
    """Identify potential features for recommendation."""
    print("\n" + "="*50)
    print("POTENTIAL FEATURES FOR RECOMMENDATION")
    print("="*50)
    
    # Load samples
    props_sample1 = pd.read_csv(Path(DATA_PATH) / "item_properties_part1.csv", nrows=200000)
    props_sample2 = pd.read_csv(Path(DATA_PATH) / "item_properties_part2.csv", nrows=200000)
    props_sample = pd.concat([props_sample1, props_sample2])
    
    # Count occurrences of each property
    property_counts = props_sample['property'].value_counts()
    
    # Determine item coverage for each property
    property_coverage = {}
    total_items = props_sample['itemid'].nunique()
    
    for prop in property_counts.index[:30]:  # Top 30 properties
        items_with_prop = props_sample[props_sample['property'] == prop]['itemid'].nunique()
        property_coverage[prop] = items_with_prop / total_items
    
    print("\nTop properties with their item coverage:")
    for prop, coverage in sorted(property_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prop}: {coverage:.2%} of items")
    
    # Important properties for e-commerce recommendation
    key_properties = [
        'categoryid',      # Item category
        'brand',           # Brand name
        'price',           # Price information
        'available',       # Availability status
        'color',           # Color information
        'size',            # Size information
        'weight',          # Weight information
        'description',     # Text description
        'title',           # Item title
        'rating',          # User ratings if available
        'timestamp'        # Temporal information
    ]
    
    print("\nRecommended features for e-commerce recommendation:")
    for prop in key_properties:
        coverage = property_coverage.get(prop, 0)
        status = "Available" if prop in property_coverage else "Not Found"
        print(f"  {prop}: {status}, Coverage: {coverage:.2%}")
    
    # Get sample values for key properties
    print("\nSample values for key properties:")
    for prop in key_properties:
        if prop in property_coverage:
            sample_values = props_sample[props_sample['property'] == prop]['value'].sample(min(5, property_counts[prop])).tolist()
            print(f"  {prop}: {sample_values}")
    
    # Clean up
    del props_sample, props_sample1, props_sample2
    gc.collect()

def provide_recommendation_strategy():
    """Provide recommendation strategy based on data analysis."""
    print("\n" + "="*50)
    print("RECOMMENDATION STRATEGY")
    print("="*50)
    
    print("""
Based on the data analysis, here's a recommended approach for building an e-commerce recommendation system:

1. HYBRID APPROACH
   - Combine collaborative filtering (from user events) with content-based filtering (from item properties)
   - Leverage category hierarchy for better generalization

2. FUNNEL-AWARE FEATURES
   - User funnel stage (view → cart → purchase)
   - Conversion rates at each stage
   - Sequential patterns from session data

3. CONTENT-BASED FEATURES
   - Category hierarchy (utilizing the full category tree)
   - Brand preferences
   - Price sensitivity
   - Other key attributes (color, size, etc.)

4. COLLABORATIVE FEATURES
   - User-item interaction matrix
   - Implicit feedback weighting (transaction > cart > view)
   - Item co-occurrence patterns
   - Session-based patterns

5. TEMPORAL FEATURES
   - Recency of interactions
   - Seasonality patterns if present
   - Time spent in each funnel stage

6. IMPLEMENTATION RECOMMENDATIONS
   - For large datasets: PyTorch or TensorFlow for distributed training
   - Consider matrix factorization with side information
   - Neural collaborative filtering with content embeddings
   - Sequence modeling with transformers for session data
   - Multi-objective optimization (conversion vs. diversity)
   
7. EVALUATION METRICS
   - Precision and Recall at k
   - NDCG for ranking quality
   - Conversion rate improvements
   - Category and brand diversity
   - Cold-start item performance
    """)

def main():
    """Main analysis function."""
    print("="*50)
    print("E-COMMERCE DATA ANALYSIS FOR RECOMMENDATION SYSTEM")
    print("="*50)
    print(f"Data path: {DATA_PATH}")
    print(f"Analysis started at: {datetime.now()}")
    
    try:
        # Analyze each dataset
        events_stats = analyze_events_data()
        item_props_stats = analyze_item_properties()
        category_stats = analyze_category_tree()
        
        # Analyze cross-dataset relationships
        analyze_cross_dataset_relationships()
        
        # Identify potential features
        identify_potential_features()
        
        # Provide recommendation strategy
        provide_recommendation_strategy()
        
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total events: {events_stats['total_events']:,}")
        print(f"Unique users: {events_stats['user_count']:,}")
        print(f"Unique items: {events_stats['item_count']:,}")
        print(f"Item properties: {item_props_stats['total_properties']:,}")
        print(f"Categories: {category_stats['total_categories']:,}")
        print(f"Analysis completed at: {datetime.now()}")
        
    except Exception as e:
        import traceback
        print(f"Error during analysis: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()