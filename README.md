# Retail Intelligence - Smart Retail Analytics

This project applies end-to-end data mining on the **UCI Online Retail** transactional dataset to produce practical retail intelligence for segmentation, cross-selling, and high-value customer prediction.

The analysis is implemented in **`Final Project.ipynb`** and covers preprocessing, exploratory analysis, similarity analysis, clustering, association-rule mining, and supervised classification.

## Project Objectives

1. Build customer-level behavioral features using **RFM** (Recency, Frequency, Monetary).
2. Measure customer similarity using financial and product-overlap metrics.
3. Segment customers with clustering for targeted marketing actions.
4. Discover frequent product co-purchase rules for bundling and recommendations.
5. Predict high-value customers using Naive Bayes and SVM models.

## Dataset

- **Source**: UCI Online Retail
- **Raw scope**: 541,909 transactions (Dec 2010 - Dec 2011)
- **Key fields**: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

### Cleaning and feature engineering

- Removed rows with missing `CustomerID`
- Removed cancellation invoices (`InvoiceNo` starting with `C`)
- Created `TotalAmount = Quantity * UnitPrice`
- Computed customer-level **RFM** features
- Added customer country and one-hot encoded country features

### Processed data snapshot (from notebook output)

- Total records: **397,924**
- Unique customers: **4,339**
- Unique products: **3,665**
- Countries: **37**
- Period: **2010-12-01 to 2011-12-09**

## Methods and Workflow

### 1) EDA and RFM distributions

Visual exploration of customer behavior distributions (recency, frequency, monetary) to understand skewness and spending concentration.

### 2) Similarity analysis

- **Euclidean distance** on scaled numeric behavior features
- **Jaccard similarity** on binary product-purchase patterns
- Heatmaps used to inspect customer-to-customer similarity structures

### 3) Clustering (KMeans + DBSCAN)

- KMeans model selection with elbow/silhouette analysis
- PCA projection used for 2D cluster visualization
- DBSCAN used to detect dense regions and noise customers

**Notebook-reported clustering metrics**
- Best K based on silhouette search: **2**
- Silhouette score at K=3: **0.594**
- KMeans (K=3) distribution:
  - Cluster 0: 3,231 customers (74.5%)
  - Cluster 1: 1,082 customers (24.9%)
  - Cluster 2: 26 customers (0.6%)
- DBSCAN (`eps=0.5`, `min_samples=5`):
  - Cluster 0: 4,285 customers (98.8%)
  - Noise points: 54 customers (1.2%)

### 4) Association rule mining (Apriori + FP-Growth)

- Built transaction basket matrix
- Generated frequent itemsets and rules
- Compared runtime of Apriori vs FP-Growth
- Evaluated rules with support, confidence, and lift

**Notebook-reported outputs**
- Apriori runtime: **0.310878 s**
- FP-Growth runtime: **2.872380 s**
- Rules with confidence >= 0.6: **4,276**

### 5) High-value customer classification

Two Naive Bayes variants were used:
- **GaussianNB** on numeric RFM features
- **BernoulliNB** on binary purchase-flag features

**Notebook-reported accuracy**
- GaussianNB: **0.96**
- BernoulliNB: **0.84**

### 6) SVM modeling with GridSearch

- Tuned linear and RBF kernels
- Evaluated CV and held-out test performance

**Notebook-reported SVM results**
- Linear SVM (best params: `C=10`, `kernel=linear`)
  - CV accuracy: **0.990**
  - Test accuracy: **0.992**
- RBF SVM (best params: `C=10`, `gamma=0.1`, `kernel=rbf`)
  - CV accuracy: **0.962**
  - Test accuracy: **0.962**

## Business Value

- Identify high-value and at-risk customer groups
- Improve campaign targeting by segment
- Support product bundling and recommendation strategies from association rules
- Prioritize retention and upsell actions using predictive scores

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- mlxtend
- openpyxl

## Repository Contents

- `Final Project.ipynb` - full analysis notebook
- `Online_Retail.xlsx` - dataset used in the notebook
- `Final Lab Project DM.docx` - project report document

## How to Run

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn mlxtend openpyxl jupyter
   ```
3. Open Jupyter:
   ```bash
   jupyter notebook
   ```
4. Run `Final Project.ipynb` from top to bottom.

