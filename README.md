# 📩 Spam Message Classification System
### Minor Project | Python · Pandas · NLTK · Scikit-learn

---

## 📁 Project Structure

```
Spam Message Classification/
│
├── spam.csv                          ← SMS Spam Collection dataset
│
├── phase1_load_explore.py            ← Phase 1 : Load & Explore Data
├── phase2_preprocess.py              ← Phase 2 : Text Preprocessing
├── phase3_feature_extraction.py      ← Phase 3 : CountVectorizer & TF-IDF
├── phase4_train_models.py            ← Phase 4 : Train Models (NB / LR / RF)
├── phase5_evaluate_visualize.py      ← Phase 5 : Evaluate & Confusion Matrices
├── spam_classification_full_project.py  ← End-to-End Script (all phases combined)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. NLTK resources
NLTK data is downloaded automatically on first run. No manual step needed.

### 3. Dataset
The `spam.csv` file is included in the repository (SMS Spam Collection Dataset).
If you need to re-download it:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## ▶️ Running Each Phase

Run phases in order from the repo root:

```bash
python phase1_load_explore.py
python phase2_preprocess.py
python phase3_feature_extraction.py
python phase4_train_models.py
python phase5_evaluate_visualize.py
```

Or run the full pipeline in one go:
```bash
python spam_classification_full_project.py
```

All scripts resolve paths relative to their own location, so they work
correctly regardless of which directory you run them from.

---

## 📊 Outputs

After running all phases the following files are generated:

| File | Description |
|---|---|
| `spam_preprocessed.csv` | Cleaned text (output of phase 2) |
| `feature_objects.pkl` | Vectorizers + feature matrices (phase 3) |
| `trained_models.pkl` | All 6 trained models (phase 4) |
| `confusion_matrices_all_models.png` | Grid of all confusion matrices |
| `confusion_matrix_*.png` | Individual confusion matrices |
| `model_comparison.png` | Bar chart comparing model metrics |
| `model_evaluation_summary.csv` | Per-model accuracy / precision / recall / F1 |
| `model_metrics_detailed.csv` | Detailed metric breakdown |
