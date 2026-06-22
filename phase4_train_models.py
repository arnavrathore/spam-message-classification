"""
PHASE 4: Train Models
Purpose: Train three classification models on both vectorization methods
Models: Multinomial Naive Bayes, Logistic Regression, Random Forest
Train/Test: 80/20 split with random_state=42
"""

import pandas as pd
import numpy as np
import re
import string
import pickle
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ============================================================================
# SETUP: Download NLTK data
# ============================================================================
print("=" * 70)
print("PHASE 4: TRAIN MODELS")
print("=" * 70)

print("\n[SETUP] Downloading NLTK data...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("✓ NLTK data ready\n")
except Exception as e:
    print(f"⚠️  Warning: {e}\n")

# ============================================================================
# STEP 1: Load Preprocessed Data
# ============================================================================
print("-" * 70)
print("STEP 1: Load Preprocessed Data")
print("-" * 70)

# Try to load preprocessed data from Phase 3, otherwise load raw and preprocess
try:
    df = pd.read_csv('spam_preprocessed.csv', encoding='utf-8')
    print("✓ Loaded: spam_preprocessed.csv (from Phase 3)")
    
    # Identify text column
    text_column = 'text' if 'text' in df.columns else 'v2_processed'
    label_column = 'label' if 'label' in df.columns else 'v1'
    
except FileNotFoundError:
    print("⚠️  spam_preprocessed.csv not found, loading raw data and preprocessing...")
    
    try:
        df = pd.read_csv('spam.csv', encoding='utf-8', usecols=[0, 1], header=None)
    except UnicodeDecodeError:
        df = pd.read_csv('spam.csv', encoding='latin1', usecols=[0, 1], header=None)
    
    df.columns = ['v1', 'v2']
    
    # Preprocess
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    
    print("  Preprocessing...", end="", flush=True)
    df['v2'] = df['v2'].apply(preprocess_text)
    print(" ✓")
    
    text_column = 'v2'
    label_column = 'v1'

# Clean data
df = df.dropna(subset=[text_column, label_column])

print(f"✓ Total messages: {len(df):,}")
print(f"  Text column: '{text_column}'")
print(f"  Label column: '{label_column}'")

# ============================================================================
# STEP 2: Encode Labels & Create Features
# ============================================================================
print("\n" + "-" * 70)
print("STEP 2: Encode Labels & Create Features")
print("-" * 70)

label_map = {"ham": 0, "spam": 1}
df['label_encoded'] = df[label_column].map(label_map)

X = df[text_column].to_numpy(dtype=object)
y = df['label_encoded'].to_numpy(dtype=int)

print(f"✓ X shape: {X.shape}")
print(f"✓ y shape: {y.shape}")

class_counts = pd.Series(y).value_counts().sort_index()
print(f"\nClass distribution:")
print(f"  Ham (0): {class_counts[0]:,} ({class_counts[0]/len(y)*100:.2f}%)")
print(f"  Spam (1): {class_counts[1]:,} ({class_counts[1]/len(y)*100:.2f}%)")

# ============================================================================
# STEP 3: Train/Test Split
# ============================================================================
print("\n" + "-" * 70)
print("STEP 3: Train/Test Split (80/20, random_state=42)")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\n✓ Training set: {len(X_train):,} messages")
print(f"  - Ham: {(y_train == 0).sum():,}")
print(f"  - Spam: {(y_train == 1).sum():,}")

print(f"\n✓ Test set: {len(X_test):,} messages")
print(f"  - Ham: {(y_test == 0).sum():,}")
print(f"  - Spam: {(y_test == 1).sum():,}")

# ============================================================================
# STEP 4: Create Feature Vectors
# ============================================================================
print("\n" + "-" * 70)
print("STEP 4: Create Feature Vectors")
print("-" * 70)

print("\nCountVectorizer:")
count_vectorizer = CountVectorizer(max_features=5000)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)
print(f"  ✓ Fitted vocabulary: {len(count_vectorizer.vocabulary_):,} terms")
print(f"  ✓ X_train shape: {X_train_count.shape}")
print(f"  ✓ X_test shape: {X_test_count.shape}")

print("\nTfidfVectorizer:")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"  ✓ Fitted vocabulary: {len(tfidf_vectorizer.vocabulary_):,} terms")
print(f"  ✓ X_train shape: {X_train_tfidf.shape}")
print(f"  ✓ X_test shape: {X_test_tfidf.shape}")

# ============================================================================
# STEP 5: Train Models
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: TRAIN MODELS")
print("=" * 70)

# Store all trained models
models = {}

# ─────────────────────────────────────────────────────────────────────────────
# 5.1: Multinomial Naive Bayes
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("5.1: Multinomial Naive Bayes")
print("-" * 70)

print("\nTraining on CountVectorizer features...")
nb_count = MultinomialNB()
nb_count.fit(X_train_count, y_train)
y_pred_nb_count = nb_count.predict(X_test_count)
acc_nb_count = accuracy_score(y_test, y_pred_nb_count)
print(f"  ✓ Test Accuracy: {acc_nb_count:.4f}")
models['NB_Count'] = (nb_count, 'CountVectorizer', y_pred_nb_count)

print("\nTraining on TfidfVectorizer features...")
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
y_pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)
acc_nb_tfidf = accuracy_score(y_test, y_pred_nb_tfidf)
print(f"  ✓ Test Accuracy: {acc_nb_tfidf:.4f}")
models['NB_TfIdf'] = (nb_tfidf, 'TfidfVectorizer', y_pred_nb_tfidf)

# ─────────────────────────────────────────────────────────────────────────────
# 5.2: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("5.2: Logistic Regression (max_iter=1000)")
print("-" * 70)

print("\nTraining on CountVectorizer features...")
lr_count = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr_count.fit(X_train_count, y_train)
y_pred_lr_count = lr_count.predict(X_test_count)
acc_lr_count = accuracy_score(y_test, y_pred_lr_count)
print(f"  ✓ Test Accuracy: {acc_lr_count:.4f}")
models['LR_Count'] = (lr_count, 'CountVectorizer', y_pred_lr_count)

print("\nTraining on TfidfVectorizer features...")
lr_tfidf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr_tfidf.fit(X_train_tfidf, y_train)
y_pred_lr_tfidf = lr_tfidf.predict(X_test_tfidf)
acc_lr_tfidf = accuracy_score(y_test, y_pred_lr_tfidf)
print(f"  ✓ Test Accuracy: {acc_lr_tfidf:.4f}")
models['LR_TfIdf'] = (lr_tfidf, 'TfidfVectorizer', y_pred_lr_tfidf)

# ─────────────────────────────────────────────────────────────────────────────
# 5.3: Random Forest
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("5.3: Random Forest (n_estimators=100)")
print("-" * 70)

print("\nTraining on CountVectorizer features...")
rf_count = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_count.fit(X_train_count, y_train)
y_pred_rf_count = rf_count.predict(X_test_count)
acc_rf_count = accuracy_score(y_test, y_pred_rf_count)
print(f"  ✓ Test Accuracy: {acc_rf_count:.4f}")
models['RF_Count'] = (rf_count, 'CountVectorizer', y_pred_rf_count)

print("\nTraining on TfidfVectorizer features...")
rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_tfidf.fit(X_train_tfidf, y_train)
y_pred_rf_tfidf = rf_tfidf.predict(X_test_tfidf)
acc_rf_tfidf = accuracy_score(y_test, y_pred_rf_tfidf)
print(f"  ✓ Test Accuracy: {acc_rf_tfidf:.4f}")
models['RF_TfIdf'] = (rf_tfidf, 'TfidfVectorizer', y_pred_rf_tfidf)

# ============================================================================
# STEP 6: Detailed Evaluation - Classification Reports
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: DETAILED EVALUATION - CLASSIFICATION REPORTS")
print("=" * 70)

for model_name, (model, vectorizer_type, y_pred) in models.items():
    print(f"\n{model_name} ({vectorizer_type})")
    print("─" * 70)
    print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM'], digits=4))

# ============================================================================
# STEP 7: Summary Comparison Table
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: MODEL PERFORMANCE SUMMARY")
print("=" * 70)

summary_data = []
for model_name, (model, vectorizer_type, y_pred) in models.items():
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    summary_data.append({
        'Model': model_name.replace('_', ' '),
        'Vectorizer': vectorizer_type,
        'Accuracy': f'{acc:.4f}',
        'Precision': f'{precision:.4f}',
        'Recall': f'{recall:.4f}',
        'F1-Score': f'{f1:.4f}'
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# ============================================================================
# STEP 8: Find Best Model
# ============================================================================
print("\n" + "-" * 70)
print("STEP 8: BEST MODEL SELECTION")
print("-" * 70)

best_f1 = -1
best_model_name = None
best_model_info = None

for model_name, (model, vectorizer_type, y_pred) in models.items():
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = model_name
        best_model_info = {
            'name': model_name,
            'vectorizer': vectorizer_type,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model,
            'predictions': y_pred
        }

print(f"\n🏆 BEST MODEL: {best_model_info['name']} ({best_model_info['vectorizer']})")
print(f"   Accuracy:  {best_model_info['accuracy']:.4f}")
print(f"   Precision: {best_model_info['precision']:.4f}")
print(f"   Recall:    {best_model_info['recall']:.4f}")
print(f"   F1-Score:  {best_model_info['f1']:.4f}")

print(f"\nWhy is this the best model?")
print(f"   → Highest F1-Score ({best_model_info['f1']:.4f})")
print(f"   → Balances Precision ({best_model_info['precision']:.4f}) and Recall ({best_model_info['recall']:.4f})")
print(f"   → Better at identifying SPAM without too many false positives")

# ============================================================================
# STEP 9: Save Models & Vectorizers
# ============================================================================
print("\n" + "-" * 70)
print("STEP 9: Save Models & Vectorizers")
print("-" * 70)

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
    'best_model': best_model_info['model'],
}

try:
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(model_objects, f)
    print("✓ Saved: trained_models.pkl")
except Exception as e:
    print(f"⚠️  Could not save models: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 4 SUMMARY")
print("=" * 70)
print(f"✓ Loaded {len(df):,} messages")
print(f"✓ Split into {len(X_train):,} train / {len(X_test):,} test (80/20)")
print(f"✓ Trained 6 models:")
print(f"   - 2x Multinomial Naive Bayes")
print(f"   - 2x Logistic Regression")
print(f"   - 2x Random Forest")
print(f"✓ Best Model: {best_model_info['name']} (F1: {best_model_info['f1']:.4f})")
print(f"✓ Models saved to trained_models.pkl")
print(f"\n→ NEXT STEP: Run Phase 5 (Visualize Results & Confusion Matrices)")
print("=" * 70)
