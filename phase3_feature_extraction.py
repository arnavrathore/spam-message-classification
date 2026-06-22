"""
PHASE 3: Feature Extraction
Purpose: Convert preprocessed text into numerical features using two vectorizers
Methods: CountVectorizer (Bag-of-Words) vs TfidfVectorizer (TF-IDF weights)
Both use max_features=5000 to limit vocabulary size
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# ============================================================================
# STEP 1: Load Preprocessed Data
# ============================================================================
print("=" * 70)
print("PHASE 3: FEATURE EXTRACTION")
print("=" * 70)

print("\n" + "-" * 70)
print("STEP 1: Load Preprocessed Data")
print("-" * 70)

# Try different possible file paths
try:
    df = pd.read_csv('spam_preprocessed.csv', encoding='utf-8')
    print("✓ Loaded: spam_preprocessed.csv")
except FileNotFoundError:
    try:
        df = pd.read_csv('data/spam_preprocessed.csv', encoding='utf-8')
        print("✓ Loaded: data/spam_preprocessed.csv")
    except FileNotFoundError:
        print("❌ File not found: spam_preprocessed.csv")
        print("   Run Phase 2 first: python phase2_preprocess.py")
        raise

print(f"  Total rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")

# Identify the text column (could be 'text', 'clean_message', or 'v2_processed')
text_column = None
if 'text' in df.columns:
    text_column = 'text'
elif 'clean_message' in df.columns:
    text_column = 'clean_message'
elif 'v2_processed' in df.columns:
    text_column = 'v2_processed'
else:
    print(f"❌ No text column found. Available columns: {list(df.columns)}")
    raise ValueError("Cannot find text column")

label_column = 'label' if 'label' in df.columns else 'v1'

print(f"  Text column: '{text_column}'")
print(f"  Label column: '{label_column}'")

# Drop any rows with missing text
initial_count = len(df)
df = df.dropna(subset=[text_column, label_column])
if len(df) < initial_count:
    print(f"  ⚠️  Dropped {initial_count - len(df)} rows with missing values")

# ============================================================================
# STEP 2: Encode Labels
# ============================================================================
print("\n" + "-" * 70)
print("STEP 2: Encode Labels")
print("-" * 70)

label_map = {"ham": 0, "spam": 1}
df['label_encoded'] = df[label_column].map(label_map)

print(f"Label encoding: ham → 0, spam → 1")
print(f"\nClass distribution:")
class_counts = df['label_encoded'].value_counts().sort_index()
for label_value, count in class_counts.items():
    label_name = "ham" if label_value == 0 else "spam"
    pct = (count / len(df)) * 100
    print(f"  {label_name} (0): {count:6,} messages ({pct:6.2f}%)")

# ============================================================================
# STEP 3: Prepare Features and Labels
# ============================================================================
print("\n" + "-" * 70)
print("STEP 3: Prepare Features (X) and Labels (y)")
print("-" * 70)

X = df[text_column].to_numpy(dtype=object)
y = df['label_encoded'].to_numpy(dtype=int)

print(f"✓ Feature vector X shape: {X.shape}")
print(f"✓ Label vector y shape: {y.shape}")
print(f"  Sample X[0][:80]: {X[0][:80]}...")

# ============================================================================
# STEP 4: Train/Test Split (80/20)
# ============================================================================
print("\n" + "-" * 70)
print("STEP 4: Train/Test Split (80/20, random_state=42)")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y  # Maintain class distribution
)

print(f"\n✓ Training set size: {len(X_train):,} messages")
print(f"  - Ham: {(y_train == 0).sum():,} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)")
print(f"  - Spam: {(y_train == 1).sum():,} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)")

print(f"\n✓ Test set size: {len(X_test):,} messages")
print(f"  - Ham: {(y_test == 0).sum():,} ({(y_test == 0).sum() / len(y_test) * 100:.2f}%)")
print(f"  - Spam: {(y_test == 1).sum():,} ({(y_test == 1).sum() / len(y_test) * 100:.2f}%)")

# ============================================================================
# STEP 5A: CountVectorizer (Bag-of-Words)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5A: CountVectorizer (Bag-of-Words)")
print("=" * 70)
print("Description: Counts raw term frequencies (how many times each word appears)")
print("-" * 70)

count_vectorizer = CountVectorizer(max_features=5000)

print("\nFitting on training data...")
X_train_count = count_vectorizer.fit_transform(X_train)

print("Transforming test data...")
X_test_count = count_vectorizer.transform(X_test)

vocab_size_count = len(count_vectorizer.vocabulary_)
print(f"\n✓ Vocabulary size: {vocab_size_count:,} unique terms")
print(f"✓ X_train_count shape: {X_train_count.shape} (rows × features)")
print(f"✓ X_test_count shape: {X_test_count.shape} (rows × features)")
print(f"  Matrix type: Sparse (CSR format) - saves memory")

# Show sparsity
sparsity_count = 1.0 - (X_train_count.nnz / (X_train_count.shape[0] * X_train_count.shape[1]))
print(f"  Sparsity: {sparsity_count * 100:.2f}% (most entries are 0)")

# Show some statistics
print(f"\nCount statistics (training set):")
print(f"  Min count per message: {X_train_count.min()}")
print(f"  Max count per message: {X_train_count.max()}")
print(f"  Mean count per message: {X_train_count.mean():.2f}")

# Show top 15 terms by feature index
print(f"\nTop 15 most frequent terms (by appearance):")
vocab_list = sorted(count_vectorizer.vocabulary_.items(), key=lambda x: x[1])[:15]
for idx, (term, feature_id) in enumerate(vocab_list, 1):
    print(f"  {idx:2d}. {term:<20s} (feature ID: {feature_id})")

# ============================================================================
# STEP 5B: TfidfVectorizer (TF-IDF Weights)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5B: TfidfVectorizer (TF-IDF Weights)")
print("=" * 70)
print("Description: Weights terms by importance (common words get lower weight)")
print("-" * 70)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)

print("\nFitting on training data...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

print("Transforming test data...")
X_test_tfidf = tfidf_vectorizer.transform(X_test)

vocab_size_tfidf = len(tfidf_vectorizer.vocabulary_)
print(f"\n✓ Vocabulary size: {vocab_size_tfidf:,} unique terms")
print(f"✓ X_train_tfidf shape: {X_train_tfidf.shape} (rows × features)")
print(f"✓ X_test_tfidf shape: {X_test_tfidf.shape} (rows × features)")
print(f"  Matrix type: Sparse (CSR format) - saves memory")

# Show sparsity
sparsity_tfidf = 1.0 - (X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))
print(f"  Sparsity: {sparsity_tfidf * 100:.2f}% (most entries are 0)")

# Show some statistics
print(f"\nTF-IDF statistics (training set):")
print(f"  Min weight: {X_train_tfidf.min():.6f}")
print(f"  Max weight: {X_train_tfidf.max():.6f}")
print(f"  Mean weight: {X_train_tfidf.mean():.6f}")

# Show top 15 terms by feature index
print(f"\nTop 15 most frequent terms (by appearance):")
vocab_list_tfidf = sorted(tfidf_vectorizer.vocabulary_.items(), key=lambda x: x[1])[:15]
for idx, (term, feature_id) in enumerate(vocab_list_tfidf, 1):
    print(f"  {idx:2d}. {term:<20s} (feature ID: {feature_id})")

# ============================================================================
# STEP 6: Side-by-Side Comparison
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Side-by-Side Comparison")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Vectorizer': ['CountVectorizer', 'TfidfVectorizer'],
    'Max Features': [5000, 5000],
    'Vocab Size': [vocab_size_count, vocab_size_tfidf],
    'X_train Shape': [str(X_train_count.shape), str(X_train_tfidf.shape)],
    'X_test Shape': [str(X_test_count.shape), str(X_test_tfidf.shape)],
    'Sparse Matrix': ['CSR', 'CSR'],
    'Value Type': ['Integer (count)', 'Float (TF-IDF)'],
    'Sparsity': [f'{sparsity_count*100:.2f}%', f'{sparsity_tfidf*100:.2f}%']
})

print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# STEP 7: Explanation & Differences
# ============================================================================
print("\n" + "-" * 70)
print("STEP 7: Key Differences Explained")
print("-" * 70)

explanation = """
CountVectorizer (Bag-of-Words):
  • Raw term frequency: counts how many times each word appears
  • Integer values (0, 1, 2, 3, ...)
  • Common words appear frequently in both ham and spam
  • Less discriminative for classification

TfidfVectorizer (Term Frequency - Inverse Document Frequency):
  • Weights words by uniqueness: rare words get higher weights
  • Formula: TF-IDF(term, doc) = TF(term, doc) × IDF(term)
  • IDF = log(total_docs / docs_containing_term)
  • Downweights common words (e.g., 'the', 'and') across all messages
  • Upweights rare words (e.g., 'winner', 'prize', 'click')
  • Better for classification: rare words are more informative

Which one to use?
  • Start with TfidfVectorizer (usually better for text classification)
  • CountVectorizer is simpler, useful as a baseline
  • Test both during model training (Phase 4)
"""
print(explanation)

# ============================================================================
# STEP 8: Sample Feature Inspection
# ============================================================================
print("-" * 70)
print("STEP 8: Sample Feature Inspection")
print("-" * 70)

# Show one example from training set
sample_idx = 0
sample_text = X_train[sample_idx]
print(f"\nSample message (index {sample_idx}):")
print(f"  Text: {sample_text[:100]}...")
print(f"  Label: {'SPAM' if y_train[sample_idx] == 1 else 'HAM'}")

# Get feature indices and values for this sample
count_row = X_train_count[sample_idx]
tfidf_row = X_train_tfidf[sample_idx]

print(f"\n  CountVectorizer: {count_row.nnz} non-zero features")
print(f"    Sample features: {dict(list(count_row.todok().items())[:5])}")

print(f"\n  TfidfVectorizer: {tfidf_row.nnz} non-zero features")
print(f"    Sample features: {dict(list(tfidf_row.todok().items())[:5])}")

# ============================================================================
# STEP 9: Save Feature Objects
# ============================================================================
print("\n" + "-" * 70)
print("STEP 9: Save Feature Objects for Phase 4")
print("-" * 70)

# Save to pickle files for use in Phase 4
feature_data = {
    'X_train_count': X_train_count,
    'X_test_count': X_test_count,
    'X_train_tfidf': X_train_tfidf,
    'X_test_tfidf': X_test_tfidf,
    'y_train': y_train,
    'y_test': y_test,
    'count_vectorizer': count_vectorizer,
    'tfidf_vectorizer': tfidf_vectorizer,
}

try:
    with open('feature_objects.pkl', 'wb') as f:
        pickle.dump(feature_data, f)
    print("✓ Saved: feature_objects.pkl")
except Exception as e:
    print(f"⚠️  Warning: Could not save pickle file: {e}")

# Also save as CSV for reference (dense format - larger file)
print("✓ Feature extraction complete")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 3 SUMMARY")
print("=" * 70)
print(f"✓ Loaded {len(df):,} preprocessed messages")
print(f"✓ Train/Test split: {len(X_train):,} train, {len(X_test):,} test (80/20)")
print(f"✓ CountVectorizer: {vocab_size_count:,} features, shape {X_train_count.shape}")
print(f"✓ TfidfVectorizer: {vocab_size_tfidf:,} features, shape {X_train_tfidf.shape}")
print(f"✓ Both matrices saved for Phase 4")
print(f"\n→ NEXT STEP: Run Phase 4 (Train Models)")
print("=" * 70)
