"""
PHASE 1: Load & Explore Data
Purpose: Load SMS Spam Collection Dataset and perform exploratory analysis
Dataset columns: v1 = label ("ham"/"spam"), v2 = message text
"""

import pandas as pd
import numpy as np

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("=" * 70)
print("PHASE 1: LOAD & EXPLORE DATA")
print("=" * 70)

# Load the CSV file with proper encoding handling
try:
    # Try UTF-8 first, then fallback to latin1 if needed
    df = pd.read_csv('spam.csv', encoding='utf-8', usecols=[0, 1])
except UnicodeDecodeError:
    print("⚠️  UTF-8 failed, trying latin1 encoding...")
    df = pd.read_csv('spam.csv', encoding='latin1', usecols=[0, 1])
except FileNotFoundError:
    print("❌ File not found: 'spam.csv' not in current directory.")
    print("   Download SMS Spam Collection Dataset and place spam.csv in the project folder.")
    raise

# Assign correct column names
df.columns = ['v1', 'v2']

# Drop any extra unnamed columns if present (common in this dataset)
if 'Unnamed: 2' in df.columns:
    df = df[['v1', 'v2']].copy()

print("\n✓ Dataset loaded successfully!")

# ============================================================================
# STEP 2: Check Dataset Shape
# ============================================================================
print("\n" + "-" * 70)
print("DATASET SHAPE")
print("-" * 70)
print(f"Total rows: {df.shape[0]:,}")
print(f"Total columns: {df.shape[1]}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# STEP 3: Check for Missing Values
# ============================================================================
print("\n" + "-" * 70)
print("MISSING VALUES CHECK")
print("-" * 70)
print(f"Null values in v1 (label): {df['v1'].isnull().sum()}")
print(f"Null values in v2 (text): {df['v2'].isnull().sum()}")

if df.isnull().sum().sum() == 0:
    print("✓ No missing values detected!")
else:
    print("⚠️  Missing values found. Removing rows with nulls...")
    df = df.dropna()
    print(f"  Rows after removal: {df.shape[0]:,}")

# ============================================================================
# STEP 4: Check Class Balance (Label Distribution)
# ============================================================================
print("\n" + "-" * 70)
print("CLASS DISTRIBUTION (Label Balance)")
print("-" * 70)
class_counts = df['v1'].value_counts()
class_percentages = (df['v1'].value_counts() / len(df) * 100).round(2)

print(f"\nLabel counts:")
for label in class_counts.index:
    count = class_counts[label]
    pct = class_percentages[label]
    bar = "█" * int(pct / 2)
    print(f"  {label:6s}: {bar:25s} {pct:6.2f}% ({count:5,} messages)")

# Check if dataset is imbalanced
if class_counts.max() / class_counts.min() > 1.5:
    print("\n⚠️  WARNING: Dataset is imbalanced (ratio > 1.5)")
    print("   Consider using class weights or resampling during model training.")
else:
    print("\n✓ Dataset is relatively balanced")

# ============================================================================
# STEP 5: Display Sample Rows
# ============================================================================
print("\n" + "-" * 70)
print("SAMPLE DATA (First 5 rows)")
print("-" * 70)
for idx in range(min(5, len(df))):
    label = df.iloc[idx]['v1']
    text = df.iloc[idx]['v2']
    text_preview = text[:70] + "..." if len(text) > 70 else text
    print(f"\n[Message {idx + 1}]")
    print(f"  Label: {label}")
    print(f"  Text: {text_preview}")

print("\n" + "-" * 70)
print("SAMPLE DATA (Random 3 rows)")
print("-" * 70)
sample_indices = np.random.choice(len(df), size=min(3, len(df)), replace=False)
for idx, sample_idx in enumerate(sample_indices, 1):
    label = df.iloc[sample_idx]['v1']
    text = df.iloc[sample_idx]['v2']
    text_preview = text[:70] + "..." if len(text) > 70 else text
    print(f"\n[Sample {idx}]")
    print(f"  Label: {label}")
    print(f"  Text: {text_preview}")

# ============================================================================
# STEP 6: Text Length Statistics
# ============================================================================
print("\n" + "-" * 70)
print("TEXT LENGTH STATISTICS")
print("-" * 70)
df['text_length'] = df['v2'].str.len()
df['word_count'] = df['v2'].str.split().str.len()

print(f"\nCharacter count per message:")
print(f"  Min: {df['text_length'].min()}")
print(f"  Max: {df['text_length'].max()}")
print(f"  Mean: {df['text_length'].mean():.2f}")
print(f"  Median: {df['text_length'].median():.2f}")
print(f"  Std Dev: {df['text_length'].std():.2f}")

print(f"\nWord count per message:")
print(f"  Min: {df['word_count'].min()}")
print(f"  Max: {df['word_count'].max()}")
print(f"  Mean: {df['word_count'].mean():.2f}")
print(f"  Median: {df['word_count'].median():.2f}")
print(f"  Std Dev: {df['word_count'].std():.2f}")

print(f"\nStatistics by label:")
stats_by_label = df.groupby('v1')[['text_length', 'word_count']].describe().round(2)
print(stats_by_label)

# ============================================================================
# STEP 7: Data Types & Basic Info
# ============================================================================
print("\n" + "-" * 70)
print("DATA TYPES & COLUMN INFO")
print("-" * 70)
print(f"v1 (label) dtype: {df['v1'].dtype}")
print(f"v2 (text) dtype: {df['v2'].dtype}")
print(f"\nUnique labels: {df['v1'].nunique()}")
print(f"Label values: {sorted(df['v1'].unique().tolist())}")

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 1 SUMMARY")
print("=" * 70)
print(f"✓ Dataset loaded: {df.shape[0]:,} messages")
print(f"✓ No missing values")
print(f"✓ Class distribution: {class_counts.get('ham', 0):,} ham, {class_counts.get('spam', 0):,} spam")
print(f"✓ Data is ready for preprocessing (Phase 2)")
print("\n→ NEXT STEP: Run Phase 2 (Preprocessing)")
print("=" * 70)
