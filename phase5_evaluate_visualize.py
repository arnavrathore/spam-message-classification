"""
PHASE 5: Evaluate & Visualize Results
Purpose: Generate detailed evaluation metrics and confusion matrix visualizations
Outputs: Classification reports, confusion matrices for all 6 models
"""

import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
    f1_score,
    roc_curve,
    auc
)

# ============================================================================
# SETUP
# ============================================================================
print("=" * 70)
print("PHASE 5: EVALUATE & VISUALIZE RESULTS")
print("=" * 70)

# Download NLTK data
print("\n[SETUP] Downloading NLTK data...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("✓ NLTK data ready\n")
except Exception as e:
    print(f"⚠️  Warning: {e}\n")

# Set up matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# STEP 1: Load Data and Preprocess
# ============================================================================
print("-" * 70)
print("STEP 1: Load Data")
print("-" * 70)

# Try to load preprocessed data
try:
    df = pd.read_csv('spam_preprocessed.csv', encoding='utf-8')
    print("✓ Loaded: spam_preprocessed.csv")
    text_column = 'text' if 'text' in df.columns else 'v2_processed'
    label_column = 'label' if 'label' in df.columns else 'v1'
except FileNotFoundError:
    print("⚠️  Loading raw data...")
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
    
    df['v2'] = df['v2'].apply(preprocess_text)
    text_column = 'v2'
    label_column = 'v1'

df = df.dropna(subset=[text_column, label_column])
print(f"✓ Total messages: {len(df):,}")

# ============================================================================
# STEP 2: Prepare Data
# ============================================================================
print("\n" + "-" * 70)
print("STEP 2: Prepare Data & Split")
print("-" * 70)

label_map = {"ham": 0, "spam": 1}
df['label_encoded'] = df[label_column].map(label_map)

X = df[text_column].to_numpy(dtype=object)
y = df['label_encoded'].to_numpy(dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"✓ Train set: {len(X_train):,} messages")
print(f"✓ Test set: {len(X_test):,} messages")

# ============================================================================
# STEP 3: Create Feature Vectors
# ============================================================================
print("\n" + "-" * 70)
print("STEP 3: Create Feature Vectors")
print("-" * 70)

count_vectorizer = CountVectorizer(max_features=5000)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"✓ CountVectorizer: {X_train_count.shape}")
print(f"✓ TfidfVectorizer: {X_train_tfidf.shape}")

# ============================================================================
# STEP 4: Train Models (or load from pickle if available)
# ============================================================================
print("\n" + "-" * 70)
print("STEP 4: Train Models")
print("-" * 70)

models_dict = {}

# Naive Bayes
print("\nTraining Naive Bayes...")
nb_count = MultinomialNB()
nb_count.fit(X_train_count, y_train)
models_dict['NB_Count'] = nb_count

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
models_dict['NB_TfIdf'] = nb_tfidf

# Logistic Regression
print("Training Logistic Regression...")
lr_count = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr_count.fit(X_train_count, y_train)
models_dict['LR_Count'] = lr_count

lr_tfidf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr_tfidf.fit(X_train_tfidf, y_train)
models_dict['LR_TfIdf'] = lr_tfidf

# Random Forest
print("Training Random Forest...")
rf_count = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_count.fit(X_train_count, y_train)
models_dict['RF_Count'] = rf_count

rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_tfidf.fit(X_train_tfidf, y_train)
models_dict['RF_TfIdf'] = rf_tfidf

print("✓ All models trained")

# ============================================================================
# STEP 5: Make Predictions
# ============================================================================
print("\n" + "-" * 70)
print("STEP 5: Make Predictions")
print("-" * 70)

# Map models to their test features
feature_maps = {
    'NB_Count': X_test_count, 'NB_TfIdf': X_test_tfidf,
    'LR_Count': X_test_count, 'LR_TfIdf': X_test_tfidf,
    'RF_Count': X_test_count, 'RF_TfIdf': X_test_tfidf,
}

predictions = {}
for model_name, model in models_dict.items():
    X_test_features = feature_maps[model_name]
    y_pred = model.predict(X_test_features)
    predictions[model_name] = y_pred
    acc = accuracy_score(y_test, y_pred)
    print(f"✓ {model_name}: Test Accuracy = {acc:.4f}")

# ============================================================================
# STEP 6: Detailed Classification Reports
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: CLASSIFICATION REPORTS (All Models)")
print("=" * 70)

for model_name, y_pred in predictions.items():
    print(f"\n{'─' * 70}")
    print(f"{model_name}")
    print(f"{'─' * 70}")
    print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM'], digits=4))

# ============================================================================
# STEP 7: Performance Summary Table
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: PERFORMANCE SUMMARY TABLE")
print("=" * 70)

summary_rows = []
for model_name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    summary_rows.append({
        'Model': model_name.replace('_', ' '),
        'Accuracy': f'{acc:.4f}',
        'Precision': f'{prec:.4f}',
        'Recall': f'{rec:.4f}',
        'F1-Score': f'{f1:.4f}',
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    })

summary_table = pd.DataFrame(summary_rows)
print("\n" + summary_table.to_string(index=False))

# ============================================================================
# STEP 8: Find Best Models
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: BEST MODELS")
print("=" * 70)

best_models = {'accuracy': None, 'f1': None}
best_scores = {'accuracy': -1, 'f1': -1}

for model_name, y_pred in predictions.items():
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    if acc > best_scores['accuracy']:
        best_scores['accuracy'] = acc
        best_models['accuracy'] = (model_name, acc)
    
    if f1 > best_scores['f1']:
        best_scores['f1'] = f1
        best_models['f1'] = (model_name, f1)

print(f"\n🏆 BEST ACCURACY: {best_models['accuracy'][0]}")
print(f"   Score: {best_models['accuracy'][1]:.4f}")

print(f"\n🏆 BEST F1-SCORE: {best_models['f1'][0]}")
print(f"   Score: {best_models['f1'][1]:.4f}")

print("\n📊 Why F1-Score matters:")
print("   F1-Score = Harmonic mean of Precision & Recall")
print("   Better than Accuracy for imbalanced datasets")
print("   Balances false positives and false negatives")

# ============================================================================
# STEP 9: Plot Confusion Matrices
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: PLOT CONFUSION MATRICES")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (model_name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['HAM', 'SPAM'],
                yticklabels=['HAM', 'SPAM'],
                ax=axes[idx], cbar=False,
                annot_kws={'size': 12, 'weight': 'bold'})
    
    # Add metrics to title
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    axes[idx].set_title(f'{model_name}\nAcc: {acc:.4f} | F1: {f1:.4f}', 
                        fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices_all_models.png")
plt.close()

# ============================================================================
# STEP 10: Individual Confusion Matrices (High Quality)
# ============================================================================
print("\n" + "-" * 70)
print("STEP 10: High-Quality Individual Confusion Matrices")
print("-" * 70)

for model_name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HAM', 'SPAM'],
                yticklabels=['HAM', 'SPAM'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'weight': 'bold'})
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {acc:.4f} | F1-Score: {f1:.4f}',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    filename = f'confusion_matrix_{model_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

# ============================================================================
# STEP 11: Model Comparison Bar Charts
# ============================================================================
print("\n" + "-" * 70)
print("STEP 11: Model Comparison Charts")
print("-" * 70)

metrics = {}
for model_name, y_pred in predictions.items():
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics[model_name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

metrics_df = pd.DataFrame(metrics).T

# Create comparison plot
fig, ax = plt.subplots(figsize=(12, 6))
metrics_df.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylim([0, 1.05])
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")
plt.close()

# ============================================================================
# STEP 12: Key Insights & Recommendations
# ============================================================================
print("\n" + "=" * 70)
print("STEP 12: KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 70)

best_f1_model = best_models['f1'][0]
best_f1_score = best_models['f1'][1]

y_pred_best = predictions[best_f1_model]
cm_best = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm_best.ravel()

print(f"""
📊 BEST MODEL: {best_f1_model} (F1: {best_f1_score:.4f})

Confusion Matrix Analysis:
  ✓ True Positives (spam correctly identified):  {tp:5d}
  ✓ True Negatives (ham correctly identified):   {tn:5d}
  ✗ False Positives (ham marked as spam):        {fp:5d}
  ✗ False Negatives (spam marked as ham):        {fn:5d}

Key Metrics:
  • Accuracy:  {accuracy_score(y_test, y_pred_best):.4f} - Overall correctness
  • Precision: {precision_score(y_test, y_pred_best, zero_division=0):.4f} - Spam detection accuracy
  • Recall:    {recall_score(y_test, y_pred_best, zero_division=0):.4f} - Spam detection coverage
  • F1-Score:  {best_f1_score:.4f} - Balanced measure

Why this model works best:
  1. High F1-Score balances precision and recall
  2. Good at catching spam ({recall_score(y_test, y_pred_best, zero_division=0):.1%} of spam detected)
  3. Minimizes false positives ({fp} legitimate messages blocked)
  4. Suitable for production use
""")

# ============================================================================
# STEP 13: Save Results to CSV
# ============================================================================
print("\n" + "-" * 70)
print("STEP 13: Save Results")
print("-" * 70)

# Save summary table
summary_table.to_csv('model_evaluation_summary.csv', index=False)
print("✓ Saved: model_evaluation_summary.csv")

# Save detailed metrics
metrics_df.to_csv('model_metrics_detailed.csv')
print("✓ Saved: model_metrics_detailed.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 5 SUMMARY")
print("=" * 70)
print(f"✓ Evaluated 6 models (3 algorithms × 2 vectorizers)")
print(f"✓ Generated confusion matrices for all models")
print(f"✓ Best Model: {best_f1_model} (F1: {best_f1_score:.4f})")
print(f"✓ Visualizations saved:")
print(f"   - confusion_matrices_all_models.png")
print(f"   - confusion_matrix_*.png (individual)")
print(f"   - model_comparison.png")
print(f"✓ Results exported to CSV files")
print(f"\n→ NEXT STEP: Review results and deploy best model")
print("=" * 70)
