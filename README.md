# 🎬 Movie Review NLP Sentiment Analysis
*Detecting **negative** reviews for The Film Junky Union with modern NLP pipelines and careful threshold tuning.*

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![pandas](https://img.shields.io/badge/pandas-EDA-green?logo=pandas)  
![numpy](https://img.shields.io/badge/numpy-Numerical-blue?logo=numpy)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)  
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-green)  
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-yellow)  
![spaCy](https://img.shields.io/badge/spaCy-NLP-8A2BE2)  
![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-orange)  
![tqdm](https://img.shields.io/badge/tqdm-ProgressBars-yellow)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Objective
Train a model to **automatically detect negative reviews** with a **minimum F1 score of 0.85** on a held-out test set.

- **Target encoding:** `0 = negative`, `1 = positive`.
- Evaluation is oriented toward the **negative class**, with **F1** as the primary metric and **probability-threshold tuning** to optimize performance on that class.

---

## 📊 Data
- **Input:** raw movie review text.
- **Target:** binary sentiment label (`0 = negative`, `1 = positive`).

---

## 🧹 NLP Pipelines
Two parallel preprocessing tracks are evaluated:

**Track A — NLTK**
- Lowercasing and regex normalization
- Tokenization and **stopword removal** (`nltk.corpus.stopwords`)
- **TF-IDF** vectorization (`TfidfVectorizer`)

**Track B — spaCy**
- Tokenization + lemmatization (`en_core_web_sm`)
- Stopword removal via spaCy vocab
- **TF-IDF** vectorization

> The notebook also performs a **threshold sweep** to study the F1–threshold relationship for the negative class and plots standard diagnostics (e.g., ROC/PR behavior).

---

## 🤖 Models Compared
- **Baseline:** `DummyClassifier` (majority class)
- **NLTK + TF-IDF + LogisticRegression** ✅ *(best generalization on test after threshold tuning)*
- **spaCy + TF-IDF + LogisticRegression**
- **spaCy + TF-IDF + LGBMClassifier**

---

## 📈 Evaluation
- **Primary metric:** **F1 (negative class)** with probability-threshold tuning
- **Additional diagnostics:** **precision**, **ROC-AUC**, **ROC/PR curves**
- **Split strategy:** `train_test_split` with a held-out test set; vectorizers fit on train to avoid leakage

**Outcome (high level):**
- **NLTK + TF-IDF + Logistic Regression** achieved the **highest F1** on the negative class and **met/exceeded 0.85** on test at a tuned threshold.
- The spaCy pipeline and **LGBMClassifier** were competitive but slightly behind the LR pipeline in final F1 on held-out data.

---

## 🧠 Why This Works
- **TF-IDF** over cleaned tokens provides a strong, sparse representation for linear models.
- **Logistic Regression** yields well-calibrated probabilities and fast inference; simple to threshold-tune for business costs.
- **Threshold tuning** lets you dial the trade-off between catching more negative reviews and avoiding false alarms.

---

## 🗂 Repo Structure
```
movie-review-nlp-sentiment/
├── notebooks/ <- Final cleaned Jupyter notebook
├── data/ <- Source data
├── requirements.txt <- Dependencies (may include conda-pinned packages)
├── LICENSE <- MIT
└── README.md <- This file
```

---

## 🚀 How to Run
1. Clone this repo:
    ```bash
    git clone https://github.com/Flamingo-Rocker/movie-review-nlp-sentiment.git
    cd movie-review-nlp-sentiment
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Install NLP resources:
    ```bash
    python -m nltk.downloader stopwords
    python -m spacy download en_core_web_sm
4. Open this notebook in `/notebooks` to reproduce this analysis.

---

## 📦 Requirements
```
pandas==2.3.2
numpy==2.2.5
numpy-base==2.2.5            
numpydoc==1.9.0            
matplotlib==3.10.5           
matplotlib-base==3.10.5           
matplotlib-inline==0.1.6        
plotly==6.3.0       
tqdm==4.67.1
scikit-learn==1.7.1
lightgbm==4.6.0
nltk==3.9.1
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
```

---

## ✅ Recommendation
- Deploy NLTK + TF-IDF + Logistic Regression at the tuned probability threshold optimized for the negative (label 0) class.
- Monitor F1 in production and periodically refresh the threshold/model as review styles and class balance drift.

---

## 🙏 Acknowledgment
Developed as part of the **TripleTen Data Science Bootcamp**, applying modern NLP preprocessing, threshold tuning, and model comparison to build a high-performing sentiment analysis model.
