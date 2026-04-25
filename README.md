# 🛒 Retail Intelligence — Smart Retail Analytics with Data Mining

> **End-to-end data mining pipeline** on the UCI Online Retail dataset covering customer segmentation, association rule mining, and high-value customer prediction.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Pipeline & Methods](#pipeline--methods)
  - [1. Data Preprocessing & EDA](#1-data-preprocessing--eda)
  - [2. Similarity Analysis](#2-similarity-analysis)
  - [3. Clustering — KMeans & DBSCAN](#3-clustering--kmeans--dbscan)
  - [4. Association Rule Mining — Apriori & FP-Growth](#4-association-rule-mining--apriori--fp-growth)
  - [5. Classification — Naive Bayes](#5-classification--naive-bayes)
  - [6. Classification — SVM with GridSearch](#6-classification--svm-with-gridsearch)
- [Key Results at a Glance](#key-results-at-a-glance)
- [Business Value](#business-value)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)

---

## Overview

This project applies a complete **data mining workflow** to the UCI Online Retail transactional dataset. Starting from raw invoice-level data, the pipeline engineers customer behavioural features, measures similarity between customers, segments them into meaningful clusters, discovers product co-purchase patterns, and finally trains machine learning classifiers to identify high-value customers automatically.

The entire analysis lives in **`Final Project.ipynb`** and is self-contained — run the notebook top-to-bottom to reproduce every result.

---

## Project Objectives

| # | Objective |
|---|-----------|
| 1 | Build customer-level **RFM** (Recency, Frequency, Monetary) features from raw transactions |
| 2 | Measure **customer similarity** using Euclidean distance and Jaccard similarity |
| 3 | **Segment customers** with KMeans and DBSCAN clustering for targeted marketing |
| 4 | Discover **frequent product co-purchase rules** (Apriori & FP-Growth) for bundling & recommendations |
| 5 | **Predict high-value customers** using Naive Bayes (Gaussian & Bernoulli) and SVM classifiers |

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail) |
| **File** | `Online_Retail.xlsx` |
| **Raw transactions** | 541,909 rows |
| **Period** | December 2010 – December 2011 |

### Key Fields

| Field | Description |
|-------|-------------|
| `InvoiceNo` | Unique invoice number (prefix `C` = cancellation) |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Units per transaction |
| `InvoiceDate` | Date and time of invoice |
| `UnitPrice` | Price per unit (GBP) |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer country |

### Data Cleaning & Feature Engineering

- ✅ Removed rows with missing `CustomerID`
- ✅ Removed cancellation invoices (`InvoiceNo` starting with `C`)
- ✅ Created `TotalAmount = Quantity × UnitPrice`
- ✅ Computed per-customer **RFM** features (Recency, Frequency, Monetary)
- ✅ Appended customer country and one-hot-encoded country columns

### Processed Dataset Snapshot

| Metric | Value |
|--------|-------|
| Clean records | **397,924** |
| Unique customers | **4,339** |
| Unique products | **3,665** |
| Countries represented | **37** |
| Date range | 2010-12-01 → 2011-12-09 |

---

## Pipeline & Methods

```
Raw Excel Data
      │
      ▼
 Preprocessing & EDA
      │
      ├──► Similarity Analysis (Euclidean + Jaccard)
      │
      ├──► Clustering (KMeans + DBSCAN)
      │
      ├──► Association Rule Mining (Apriori + FP-Growth)
      │
      └──► Classification (Naive Bayes + SVM)
```

---

### 1. Data Preprocessing & EDA

- Loaded and cleaned the raw Excel file
- Engineered `TotalAmount` and RFM customer features
- Visualised distributions of Recency, Frequency, and Monetary values
- Identified heavy right-skew in spending (most customers are occasional buyers; a small group are high-value)

---

### 2. Similarity Analysis

| Method | Features Used | Visualization |
|--------|--------------|---------------|
| **Euclidean Distance** | Scaled numeric RFM features | Heatmap |
| **Jaccard Similarity** | Binary product-purchase matrix | Heatmap |

Similarity heatmaps revealed natural groupings and confirmed that a small subset of customers share very distinct purchasing patterns from the majority.

---

### 3. Clustering — KMeans & DBSCAN

#### KMeans

- **Elbow method** and **Silhouette analysis** used for K selection
- PCA (2 components) applied for 2-D cluster visualization

| Setting | Value |
|---------|-------|
| Best K (silhouette search) | **2** |
| Silhouette score at K = 3 | **0.594** |

**KMeans (K = 3) cluster distribution:**

| Cluster | Customers | Share |
|---------|-----------|-------|
| 0 | 3,231 | 74.5 % |
| 1 | 1,082 | 24.9 % |
| 2 | 26 | 0.6 % |

#### DBSCAN (`eps = 0.5`, `min_samples = 5`)

| Label | Customers | Share |
|-------|-----------|-------|
| Cluster 0 | 4,285 | 98.8 % |
| Noise | 54 | 1.2 % |

> DBSCAN effectively identified the tiny group of outlier customers (noise) that deviate from the dense majority — useful for anomaly detection.

---

### 4. Association Rule Mining — Apriori & FP-Growth

- Built a **binary transaction basket matrix** (customer × product)
- Generated frequent itemsets and derived association rules
- Evaluated rules by **support**, **confidence**, and **lift**
- Benchmarked runtime of both algorithms

| Algorithm | Runtime |
|-----------|---------|
| Apriori | **0.310878 s** |
| FP-Growth | **2.872380 s** |
| Rules with confidence ≥ 0.6 | **4,276** |

> The high-confidence rules can directly power a **product recommendation engine** or inform promotional bundle strategies.

---

### 5. Classification — Naive Bayes

Two Naive Bayes variants tested for high-value customer prediction:

| Variant | Features | Accuracy |
|---------|----------|----------|
| **GaussianNB** | Numeric RFM features | **0.96** |
| **BernoulliNB** | Binary product-purchase flags | **0.84** |

GaussianNB significantly outperforms BernoulliNB, confirming that continuous RFM signals carry richer discriminative information than binary purchase indicators.

---

### 6. Classification — SVM with GridSearch

Both **Linear** and **RBF** kernel SVMs tuned via `GridSearchCV`:

| Kernel | Best Params | CV Accuracy | Test Accuracy |
|--------|------------|-------------|---------------|
| **Linear** | `C=10, kernel=linear` | **0.990** | **0.992** |
| **RBF** | `C=10, gamma=0.1, kernel=rbf` | **0.962** | **0.962** |

The Linear SVM achieves near-perfect accuracy (**99.2 %**) on the held-out test set, making it the top-performing model in this project.

---

## Key Results at a Glance

| Task | Best Model | Best Score |
|------|-----------|------------|
| Clustering quality | KMeans K=3 | Silhouette = 0.594 |
| Association rules | FP-Growth / Apriori | 4,276 rules @ conf ≥ 0.6 |
| Classification | Linear SVM | Test Accuracy = **99.2 %** |

---

## Business Value

| Use Case | How This Project Helps |
|----------|----------------------|
| 🎯 **Targeted Marketing** | Cluster labels enable personalised campaigns per segment |
| 💎 **VIP Identification** | SVM/NB models predict high-value customers before they churn |
| 🛍️ **Product Bundling** | Association rules reveal which products are frequently bought together |
| 🔁 **Retention** | Recency scores highlight at-risk customers needing re-engagement |
| 📈 **Upselling** | Lift scores from rules surface upgrade / add-on opportunities |

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data handling** | `pandas`, `numpy`, `openpyxl` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` |
| **Association Mining** | `mlxtend` |
| **Notebook** | `Jupyter Notebook` |

---

## Repository Structure

```
project/
├── Final Project.ipynb        # Main analysis notebook (run this)
├── Online_Retail.xlsx         # Raw UCI Online Retail dataset
├── Final Lab Project DM.docx  # Project report / documentation
└── README.md                  # This file
```

---

## How to Run

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

1. **Clone / download** this repository.

2. **Create and activate a virtual environment** *(recommended)*:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn mlxtend openpyxl jupyter
   ```

4. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

5. Open **`Final Project.ipynb`** and run all cells from top to bottom (`Kernel → Restart & Run All`).

> ⚠️ The dataset file `Online_Retail.xlsx` (~23 MB) must be present in the same directory as the notebook, or the data-loading cell will fail.

---

*Data Mining Final Project — UCI Online Retail Dataset*
