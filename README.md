# 📩 Spam Message Classification System
### Minor Project | Python · Pandas · NLTK · Scikit-learn

---

## 📁 Project Structure

```
Spam Message Classification/
│
├── data/
│   └── spam.csv                  ← Place your SMS Spam Collection dataset here
│
├── phase1_load_explore.py        ← Phase 1 : Load & Explore Data
├── phase2_preprocess.py          ← Phase 2 : Text Preprocessing
├── phase3_feature_extraction.py  ← Phase 3 : CountVectorizer & TF-IDF
├── phase4_train_models.py        ← Phase 4 : Train Models (NB / LR / RF)
├── phase5_evaluate.py            ← Phase 5 : Evaluate & Confusion Matrices
├── full_project.py               ← End-to-End Script (all phases combined)
│
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install pandas scikit-learn nltk seaborn matplotlib
```

### 2. Download NLTK resources (run once)
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 3. Dataset
Download the **SMS Spam Collection Dataset** from Kaggle:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Place `spam.csv` inside the `data/` folder.

---

## ▶️ Running Each Phase
```bash
python phase1_load_explore.py
python phase2_preprocess.py
python phase3_feature_extraction.py
python phase4_train_models.py
python phase5_evaluate.py
```

Or run the full pipeline:
```bash
python full_project.py
```
