"""
PHASE 2: Preprocess Text Data
Purpose: Clean and normalize text for feature extraction
Steps: lowercase → remove punctuation → tokenize → remove stopwords → stem with PorterStemmer

⚠️ IMPORTANT: Before running this script, execute these commands in your terminal:
   python -m nltk.downloader stopwords punkt
   OR in Python: import nltk; nltk.download('stopwords'); nltk.download('punkt')
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

# ============================================================================
# SETUP: Download required NLTK data
# ============================================================================
print("=" * 70)
print("PHASE 2: PREPROCESS TEXT DATA")
print("=" * 70)

print("\n[SETUP] Downloading NLTK data (stopwords, punkt)...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("✓ NLTK data downloaded/verified\n")
except Exception as e:
    print(f"⚠️  Error downloading NLTK data: {e}")
    print("   Run manually: python -m nltk.downloader stopwords punkt")

# ============================================================================
# STEP 1: Load Data from Phase 1
# ============================================================================
print("-" * 70)
print("STEP 1: Load Raw Data")
print("-" * 70)

try:
    df = pd.read_csv('spam.csv', encoding='utf-8', usecols=[0, 1])
except UnicodeDecodeError:
    df = pd.read_csv('spam.csv', encoding='latin1', usecols=[0, 1])

df.columns = ['v1', 'v2']

# Drop extra columns if present
if 'Unnamed: 2' in df.columns:
    df = df[['v1', 'v2']].copy()

print(f"✓ Loaded {len(df):,} messages")
print(f"  Columns: {list(df.columns)}")

# ============================================================================
# STEP 2: Initialize Preprocessing Components
# ============================================================================
print("\n" + "-" * 70)
print("STEP 2: Initialize Preprocessing Components")
print("-" * 70)

# Get English stopwords
stop_words = set(stopwords.words('english'))
print(f"✓ Loaded {len(stop_words)} English stopwords")

# Initialize Porter Stemmer
stemmer = PorterStemmer()
print(f"✓ PorterStemmer initialized")

# ============================================================================
# STEP 3: Define Preprocessing Function
# ============================================================================
print("\n" + "-" * 70)
print("STEP 3: Define Preprocessing Function")
print("-" * 70)

def preprocess_text(text):
    """
    Preprocess a single text message through all cleaning steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Tokenize into words
    4. Remove stopwords
    5. Stem words using Porter Stemmer
    
    Args:
        text (str): Raw message text
    
    Returns:
        str: Space-separated preprocessed words
    """
    
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation and special characters
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Step 3: Tokenize (split into words)
    tokens = word_tokenize(text)
    
    # Step 4: Remove stopwords and short words (len < 2)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    # Step 5: Apply Porter Stemmer
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Return as space-separated string
    return ' '.join(tokens)

print("✓ Preprocessing function defined")

# ============================================================================
# STEP 4: Show Preprocessing Examples
# ============================================================================
print("\n" + "-" * 70)
print("STEP 4: Preprocessing Examples (Before & After)")
print("-" * 70)

# Show 3 random examples
np.random.seed(42)
example_indices = np.random.choice(len(df), size=3, replace=False)

for idx, sample_idx in enumerate(example_indices, 1):
    raw_text = df.iloc[sample_idx]['v2']
    label = df.iloc[sample_idx]['v1']
    processed_text = preprocess_text(raw_text)
    
    print(f"\n[Example {idx}] Label: {label}")
    print(f"  BEFORE: {raw_text[:80]}...")
    print(f"  AFTER:  {processed_text[:80]}...")

# ============================================================================
# STEP 5: Apply Preprocessing to All Messages
# ============================================================================
print("\n" + "-" * 70)
print("STEP 5: Applying Preprocessing to All Messages")
print("-" * 70)

print("Processing...", end="", flush=True)

# Apply preprocessing function to each message
df['v2_processed'] = df['v2'].apply(preprocess_text)

print("\r✓ Processing 100% complete!")
print(f"  Total messages processed: {len(df):,}")

# ============================================================================
# STEP 6: Verify Preprocessing Results
# ============================================================================
print("\n" + "-" * 70)
print("STEP 6: Preprocessing Results Verification")
print("-" * 70)

# Check for any empty messages after preprocessing
empty_count = (df['v2_processed'].str.len() == 0).sum()
print(f"\nMessages with empty text after preprocessing: {empty_count}")

if empty_count > 0:
    print("⚠️  Removing messages with no valid words...")
    df = df[df['v2_processed'].str.len() > 0].copy()
    print(f"   Rows after removal: {len(df):,}")

# ============================================================================
# STEP 7: Statistics on Preprocessed Text
# ============================================================================
print("\n" + "-" * 70)
print("STEP 7: Preprocessed Text Statistics")
print("-" * 70)

df['processed_length'] = df['v2_processed'].str.len()
df['processed_word_count'] = df['v2_processed'].str.split().str.len()

print(f"\nProcessed message length (characters):")
print(f"  Min: {df['processed_length'].min()}")
print(f"  Max: {df['processed_length'].max()}")
print(f"  Mean: {df['processed_length'].mean():.2f}")
print(f"  Median: {df['processed_length'].median():.2f}")

print(f"\nProcessed message word count:")
print(f"  Min: {df['processed_word_count'].min()}")
print(f"  Max: {df['processed_word_count'].max()}")
print(f"  Mean: {df['processed_word_count'].mean():.2f}")
print(f"  Median: {df['processed_word_count'].median():.2f}")

print(f"\nComparison by label:")
comparison = df.groupby('v1')[['processed_length', 'processed_word_count']].agg(['min', 'mean', 'max']).round(2)
print(comparison)

# ============================================================================
# STEP 8: Display Sample Preprocessed Messages
# ============================================================================
print("\n" + "-" * 70)
print("STEP 8: Sample Preprocessed Messages")
print("-" * 70)

# Show 5 random samples with original and processed text
sample_indices = np.random.choice(len(df), size=min(5, len(df)), replace=False)

for idx, sample_idx in enumerate(sample_indices, 1):
    label = df.iloc[sample_idx]['v1']
    original = df.iloc[sample_idx]['v2']
    processed = df.iloc[sample_idx]['v2_processed']
    
    print(f"\n[Sample {idx}] Label: {label}")
    print(f"  Original : {original[:70]}{'...' if len(original) > 70 else ''}")
    print(f"  Processed: {processed[:70]}{'...' if len(processed) > 70 else ''}")

# ============================================================================
# STEP 9: Class Balance After Preprocessing
# ============================================================================
print("\n" + "-" * 70)
print("STEP 9: Class Distribution After Preprocessing")
print("-" * 70)

class_counts = df['v1'].value_counts()
class_pct = (df['v1'].value_counts() / len(df) * 100).round(2)

print(f"\n{'Label':<10} {'Count':<15} {'Percentage':<10}")
print("-" * 35)
for label in class_counts.index:
    count = class_counts[label]
    pct = class_pct[label]
    print(f"{label:<10} {count:<15,} {pct:<10.2f}%")

print(f"\nTotal messages: {len(df):,}")

# ============================================================================
# STEP 10: Most Common Words in Spam vs Ham
# ============================================================================
print("\n" + "-" * 70)
print("STEP 10: Most Common Words (Spam vs Ham)")
print("-" * 70)

def get_top_words(messages, n=10):
    """Extract and count all words from a group of messages"""
    all_words = ' '.join(messages).split()
    word_counts = Counter(all_words)
    return word_counts.most_common(n)

spam_messages = df[df['v1'] == 'spam']['v2_processed']
ham_messages = df[df['v1'] == 'ham']['v2_processed']

print("\nTop 10 words in SPAM messages:")
spam_words = get_top_words(spam_messages, n=10)
for i, (word, count) in enumerate(spam_words, 1):
    print(f"  {i:2d}. {word:<15s} ({count:6,} occurrences)")

print("\nTop 10 words in HAM messages:")
ham_words = get_top_words(ham_messages, n=10)
for i, (word, count) in enumerate(ham_words, 1):
    print(f"  {i:2d}. {word:<15s} ({count:6,} occurrences)")

# ============================================================================
# STEP 11: Save Preprocessed Data
# ============================================================================
print("\n" + "-" * 70)
print("STEP 11: Save Preprocessed Data")
print("-" * 70)

# Keep only the necessary columns for next phase
df_processed = df[['v1', 'v2_processed']].copy()
df_processed.columns = ['label', 'text']

# Save to CSV for use in Phase 3
df_processed.to_csv('spam_preprocessed.csv', index=False, encoding='utf-8')
print("✓ Preprocessed data saved to 'spam_preprocessed.csv'")

# Also save the full dataframe with both original and processed for reference
df_with_both = df[['v1', 'v2', 'v2_processed']].copy()
df_with_both.columns = ['label', 'text_original', 'text_processed']
df_with_both.to_csv('spam_with_original.csv', index=False, encoding='utf-8')
print("✓ Original + processed data saved to 'spam_with_original.csv' (for reference)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 2 SUMMARY")
print("=" * 70)
print(f"✓ Loaded {len(df):,} raw messages")
print(f"✓ Applied preprocessing: lowercase → remove punct. → tokenize → stopword removal → stem")
print(f"✓ Removed {empty_count} messages with no valid words")
print(f"✓ Final dataset: {len(df_processed):,} messages")
print(f"✓ Class distribution: {class_counts.get('ham', 0):,} ham, {class_counts.get('spam', 0):,} spam")
print(f"✓ Output files created:")
print(f"   - spam_preprocessed.csv (for Phase 3)")
print(f"   - spam_with_original.csv (reference)")
print(f"\n→ NEXT STEP: Run Phase 3 (Feature Extraction)")
print("=" * 70)
