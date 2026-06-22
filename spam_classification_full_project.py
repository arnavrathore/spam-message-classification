"""
SPAM MESSAGE CLASSIFICATION - FULL PROJECT
Complete end-to-end pipeline in one script
Phases: Load → Preprocess → Feature Extract → Train → Evaluate
"""

import pandas as pd
import numpy as np
import re
import string
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 5000
RF_N_ESTIMATORS = 100

# Download NLTK data
print("Downloading NLTK resources...")
for resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        print(f"  Warning: Could not download {resource}: {e}")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Set up visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PHASE 1: LOAD & EXPLORE DATA
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 1: LOAD & EXPLORE DATA")
print("=" * 70)

print("\nLoading dataset...")
try:
    df = pd.read_csv('spam.csv', encoding='utf-8', usecols=[0, 1], header=None)
except UnicodeDecodeError:
    df = pd.read_csv('spam.csv', encoding='latin1', usecols=[0, 1], header=None)
except FileNotFoundError:
    print("❌ Error: spam.csv not found in current directory")
    raise

# Assign column names
df.columns = ['v1', 'v2']

print(f"✓ Loaded {len(df):,} messages")
print(f"  Dataset shape: {df.shape}")

# Check for missing values
print(f"\nMissing values:")
print(f"  v1 (label): {df['v1'].isnull().sum()}")
print(f"  v2 (text): {df['v2'].isnull().sum()}")

# Class distribution
class_counts = df['v1'].value_counts()
class_pct = (df['v1'].value_counts() / len(df) * 100).round(2)

print(f"\nClass distribution:")
for label in class_counts.index:
    count = class_counts[label]
    pct = class_pct[label]
    bar = "█" * int(pct / 2)
    print(f"  {label:6s}: {bar:25s} {pct:6.2f}% ({count:5,} messages)")

# Sample messages
print(f"\nSample messages:")
for idx in range(min(3, len(df))):
    label = df.iloc[idx]['v1']
    text = df.iloc[idx]['v2'][:70] + "..." if len(df.iloc[idx]['v2']) > 70 else df.iloc[idx]['v2']
    print(f"  [{idx + 1}] {label}: {text}")

# ============================================================================
# PHASE 2: PREPROCESS TEXT DATA
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 2: PREPROCESS TEXT DATA")
print("=" * 70)

def preprocess_text(text):
    """
    Pipeline: lowercase → remove punctuation → tokenize → 
              remove stopwords → stem
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

print("\nApplying preprocessing pipeline...")
print("  (lowercase → remove punctuation → tokenize → stopwords → stem)")

df['v2_processed'] = df['v2'].apply(preprocess_text)

# Check for empty messages
empty_count = (df['v2_processed'].str.len() == 0).sum()
if empty_count > 0:
    print(f"  Removing {empty_count} messages with no valid words...")
    df = df[df['v2_processed'].str.len() > 0].copy()

print(f"✓ Preprocessing complete: {len(df):,} messages")

# Text statistics
df['text_length'] = df['v2_processed'].str.len()
df['word_count'] = df['v2_processed'].str.split().str.len()

print(f"\nPreprocessed text statistics:")
print(f"  Character count: min={df['text_length'].min()}, max={df['text_length'].max()}, mean={df['text_length'].mean():.2f}")
print(f"  Word count: min={df['word_count'].min()}, max={df['word_count'].max()}, mean={df['word_count'].mean():.2f}")

# Top words in spam vs ham
def get_top_words(messages, n=5):
    all_words = ' '.join(messages).split()
    return Counter(all_words).most_common(n)

spam_msgs = df[df['v1'] == 'spam']['v2_processed']
ham_msgs = df[df['v1'] == 'ham']['v2_processed']

print(f"\nTop 5 words in SPAM messages:")
for i, (word, count) in enumerate(get_top_words(spam_msgs, n=5), 1):
    print(f"  {i}. {word} ({count})")

print(f"\nTop 5 words in HAM messages:")
for i, (word, count) in enumerate(get_top_words(ham_msgs, n=5), 1):
    print(f"  {i}. {word} ({count})")

# ============================================================================
# PHASE 3: FEATURE EXTRACTION
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 3: FEATURE EXTRACTION")
print("=" * 70)

# Encode labels
label_map = {"ham": 0, "spam": 1}
df['label_encoded'] = df['v1'].map(label_map)

# Drop rows with missing labels
initial_count = len(df)
df = df.dropna(subset=['label_encoded'])
if len(df) < initial_count:
    print(f"  ⚠️  Dropped {initial_count - len(df)} rows with missing labels")

# Convert to numpy arrays (not lists) for sklearn compatibility
X = df['v2_processed'].to_numpy(dtype=object)
y = df['label_encoded'].to_numpy(dtype=int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain/Test split (80/20):")
print(f"  Training set: {len(X_train):,} messages")
print(f"    - Ham: {(y_train == 0).sum():,}")
print(f"    - Spam: {(y_train == 1).sum():,}")
print(f"  Test set: {len(X_test):,} messages")
print(f"    - Ham: {(y_test == 0).sum():,}")
print(f"    - Spam: {(y_test == 1).sum():,}")

# Vectorization
print(f"\nCreating feature vectors...")

print(f"\n  CountVectorizer (Bag-of-Words):")
count_vectorizer = CountVectorizer(max_features=MAX_FEATURES)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)
print(f"    ✓ Vocabulary: {len(count_vectorizer.vocabulary_):,} terms")
print(f"    ✓ X_train shape: {X_train_count.shape}")
print(f"    ✓ X_test shape: {X_test_count.shape}")

print(f"\n  TfidfVectorizer (TF-IDF weights):")
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"    ✓ Vocabulary: {len(tfidf_vectorizer.vocabulary_):,} terms")
print(f"    ✓ X_train shape: {X_train_tfidf.shape}")
print(f"    ✓ X_test shape: {X_test_tfidf.shape}")

# ============================================================================
# PHASE 4: TRAIN MODELS
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 4: TRAIN MODELS")
print("=" * 70)

models = {}

print(f"\nTraining Multinomial Naive Bayes...")
nb_count = MultinomialNB()
nb_count.fit(X_train_count, y_train)
models['NB_Count'] = (nb_count, X_test_count)

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
models['NB_TfIdf'] = (nb_tfidf, X_test_tfidf)

print(f"  ✓ Naive Bayes trained on both vectorizers")

print(f"\nTraining Logistic Regression (max_iter=1000)...")
lr_count = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs')
lr_count.fit(X_train_count, y_train)
models['LR_Count'] = (lr_count, X_test_count)

lr_tfidf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs')
lr_tfidf.fit(X_train_tfidf, y_train)
models['LR_TfIdf'] = (lr_tfidf, X_test_tfidf)

print(f"  ✓ Logistic Regression trained on both vectorizers")

print(f"\nTraining Random Forest (n_estimators={RF_N_ESTIMATORS})...")
rf_count = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
rf_count.fit(X_train_count, y_train)
models['RF_Count'] = (rf_count, X_test_count)

rf_tfidf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
rf_tfidf.fit(X_train_tfidf, y_train)
models['RF_TfIdf'] = (rf_tfidf, X_test_tfidf)

print(f"  ✓ Random Forest trained on both vectorizers")

# ============================================================================
# PHASE 5: EVALUATE & VISUALIZE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 5: EVALUATE & VISUALIZE RESULTS")
print("=" * 70)

# Make predictions and collect metrics
predictions = {}
metrics_data = []

for model_name, (model, X_test_features) in models.items():
    y_pred = model.predict(X_test_features)
    predictions[model_name] = y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics_data.append({
        'Model': model_name.replace('_', ' '),
        'Accuracy': f'{acc:.4f}',
        'Precision': f'{prec:.4f}',
        'Recall': f'{rec:.4f}',
        'F1-Score': f'{f1:.4f}'
    })

print(f"\nModel Performance Summary:")
print(f"{'─' * 70}")
metrics_df = pd.DataFrame(metrics_data)
print(metrics_df.to_string(index=False))

# Find best model by F1
best_f1 = -1
best_model_name = None
best_metrics = None

for model_name, y_pred in predictions.items():
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = model_name
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        best_metrics = {
            'name': model_name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'predictions': y_pred
        }

# Print best model details
print(f"\n{'=' * 70}")
print(f"🏆 BEST MODEL: {best_model_name} (F1-Score: {best_metrics['f1']:.4f})")
print(f"{'=' * 70}")

print(f"\nDetailed Metrics:")
print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
print(f"  Precision: {best_metrics['precision']:.4f}")
print(f"  Recall:    {best_metrics['recall']:.4f}")
print(f"  F1-Score:  {best_metrics['f1']:.4f}")

print(f"\nConfusion Matrix:")
print(f"                    Predicted")
print(f"                 HAM      SPAM")
print(f"  Actual  HAM   {best_metrics['tn']:6d}   {best_metrics['fp']:6d}")
print(f"          SPAM  {best_metrics['fn']:6d}   {best_metrics['tp']:6d}")

print(f"\nInterpretation:")
print(f"  ✓ True Positives (Spam caught):      {best_metrics['tp']:6d}")
print(f"  ✓ True Negatives (Ham passed):       {best_metrics['tn']:6d}")
print(f"  ✗ False Positives (Ham blocked):     {best_metrics['fp']:6d}")
print(f"  ✗ False Negatives (Spam missed):     {best_metrics['fn']:6d}")

# Classification report for best model
print(f"\nClassification Report ({best_model_name}):")
print(f"{'─' * 70}")
print(classification_report(y_test, best_metrics['predictions'], 
                          target_names=['HAM', 'SPAM'], digits=4))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print(f"\nGenerating visualizations...")

# 1. Confusion matrices for all models (grid)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (model_name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HAM', 'SPAM'],
                yticklabels=['HAM', 'SPAM'],
                ax=axes[idx], cbar=False,
                annot_kws={'size': 12, 'weight': 'bold'})
    
    axes[idx].set_title(f'{model_name}\nAcc: {acc:.4f} | F1: {f1:.4f}',
                       fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('01_confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 01_confusion_matrices_all_models.png")
plt.close()

# 2. Confusion matrix for best model (large)
fig, ax = plt.subplots(figsize=(8, 6))
cm_best = confusion_matrix(y_test, best_metrics['predictions'])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
            xticklabels=['HAM', 'SPAM'],
            yticklabels=['HAM', 'SPAM'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 14, 'weight': 'bold'},
            ax=ax)
ax.set_title(f'{best_model_name} - Confusion Matrix\nAccuracy: {best_metrics["accuracy"]:.4f} | F1: {best_metrics["f1"]:.4f}',
            fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('02_confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 02_confusion_matrix_best_model.png")
plt.close()

# 3. Model comparison bar chart
metrics_comparison = {}
for model_name, y_pred in predictions.items():
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    metrics_comparison[model_name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1
    }

metrics_comp_df = pd.DataFrame(metrics_comparison).T

fig, ax = plt.subplots(figsize=(12, 6))
metrics_comp_df.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylim([0, 1.05])
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('03_model_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 03_model_comparison.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print(f"\nSaving results...")

# Save metrics to CSV
metrics_df.to_csv('model_evaluation_summary.csv', index=False)
print(f"  ✓ Saved: model_evaluation_summary.csv")

metrics_comp_df.to_csv('model_metrics_detailed.csv')
print(f"  ✓ Saved: model_metrics_detailed.csv")

# Save models to pickle
model_objects = {
    'nb_count': nb_count,
    'nb_tfidf': nb_tfidf,
    'lr_count': lr_count,
    'lr_tfidf': lr_tfidf,
    'rf_count': rf_count,
    'rf_tfidf': rf_tfidf,
    'count_vectorizer': count_vectorizer,
    'tfidf_vectorizer': tfidf_vectorizer,
    'best_model_name': best_model_name,
}

with open('trained_models.pkl', 'wb') as f:
    pickle.dump(model_objects, f)
print(f"  ✓ Saved: trained_models.pkl")

# ============================================================================
# PROJECT COMPLETION SUMMARY
# ============================================================================
print(f"\n{'=' * 70}")
print(f"PROJECT COMPLETE ✓")
print(f"{'=' * 70}")

print(f"""
📊 PROJECT SUMMARY

Data:
  • Total messages: {len(df):,}
  • Train set: {len(X_train):,} messages
  • Test set: {len(X_test):,} messages
  • Class distribution: {class_counts['ham']:,} ham, {class_counts['spam']:,} spam

Preprocessing:
  • Pipeline: lowercase → remove punct. → tokenize → stopwords → stem
  • Empty messages removed: {empty_count}
  • Final messages: {len(df):,}

Feature Extraction:
  • CountVectorizer: {len(count_vectorizer.vocabulary_):,} features
  • TfidfVectorizer: {len(tfidf_vectorizer.vocabulary_):,} features

Models Trained:
  • Multinomial Naive Bayes (×2)
  • Logistic Regression (×2)
  • Random Forest (×2)

🏆 BEST MODEL: {best_model_name}
  • Accuracy:  {best_metrics['accuracy']:.4f}
  • Precision: {best_metrics['precision']:.4f} (avoid false positives)
  • Recall:    {best_metrics['recall']:.4f} (catch spam)
  • F1-Score:  {best_metrics['f1']:.4f} (balanced metric)

✓ Outputs Generated:
  • 01_confusion_matrices_all_models.png
  • 02_confusion_matrix_best_model.png
  • 03_model_comparison.png
  • model_evaluation_summary.csv
  • model_metrics_detailed.csv
  • trained_models.pkl

📌 Key Insights:
  • Best model balances precision and recall (F1-Score)
  • {best_metrics['tp']} spam messages correctly identified
  • {best_metrics['fp']} legitimate messages marked as spam
  • {best_metrics['fn']} spam messages missed

✅ Ready for deployment!
{'=' * 70}
""")
