# Taiwanese Bankruptcy Prediction Analysis

**Data Mining & Machine Learning Project** *EDHEC Business School - Dual Degree in digital engineering and finance

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=flat&logo=scikit-learn)
![NumPy](https://img.shields.io/badge/Library-NumPy-blue?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-lightgrey?style=flat&logo=matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Library-Seaborn-blue?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458?style=flat&logo=pandas&logoColor=white)

## Project Overview

This project applies macine learning techniques to predict and analyze corporate bankruptcy using the **Taiwanese Bankruptcy Prediction dataset**. The primary challenge of this analysis was dealing with a **highly imbalanced dataset**, where only 3.2% of companies were labeled as bankrupt. Each code is predede by a markdown cell that explain my thought when writing it. It has been realised as an individual assignment for my grade of Data Mining & Machine Learning course at EDHEC Business School.

The workflow combines **Supervised Learning** to predict risk scores and **Unsupervised Learning** to discover hidden financial profiles within the market.

---

## Disclaimer

> **Note:** As this was my very first project in Machine Learning, the primary focus was on understanding the core concepts and methodologies (discovery phase). I am aware that there is still significant room for improvement regarding code optimization, modularity, and visual presentation.

---

## Project Requirements

The assignment required analyzing a dataset of our choice under the following constraints:

* **Dataset Selection:**
    * Must be publicly available (e.g., UCI, Kaggle).
    * **Minimum Size:** At least 5,000 instances (rows).
    * **Dimensionality:** At least 15 features (variables).

* **Analysis Tasks:**
    * Perform a complete **Descriptive Analysis** (Univariate & Bivariate).
    * Apply at least **two different Machine Learning techniques** (chosen from Classification, Regression, or Clustering).

* **Methodological Requirements:**
    * Justify the choice of dataset and methods.
    * Apply **Feature Scaling** (Standardization/Normalization) where appropriate.
    * Implement **Cross-Validation** to prevent overfitting.
  
---

## The Dataset

* **Source:** [UCI Machine Learning Repository - Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)
* **Instances:** 6,819 companies
* **Features:** 95 financial ratios (e.g., ROA, Debt Ratio, Net Income to Assets)
* **Target:** Binary (`1` = Bankrupt, `0` = Solvent)
* **Imbalance:** 96.8% Solvent vs. 3.2% Bankrupt

---

## Methodology

### 1. Data Cleaning & EDA
* Removed constant columns (zero and almost zero variance) to reduce noise.
* Performed univariate and bivariate analysis to identify key risk drivers.
* Applied **StandardScaler** to normalize the 94 financial features.

### 2. Classification (Supervised Learning)
* **Model:** Logistic Regression.
* **Validation:** 5-Fold Stratified Cross-Validation on an 80/20 split.
* **Metric:** Optimized for **ROC-AUC** due to class imbalance, also F1 Score, Recall and Precision.
* **Feature Selection:** Implemented **Lasso Regression (L1)** which successfully eliminated 35 redundant features, identifying 58 not-redondant financial indicators.

### 3. Clustering (Unsupervised Learning)
* **Algorithm:** K-Means Clustering on the full dataset.
* **Optimization:** Used the **Elbow Method** to determine the optimal number of clusters ($K=6$).
* **Profiling:** Analyzed the mean financial profiles of each cluster to identify distinct market archetypes (e.g., "High-Debt/High-Risk" vs. "High-Equity/Healthy").

---

## Key Results & Findings

### Classification Performance
The Logistic Regression model demonstrated good stability (**CV Std Dev: 0.0263**) and ranking capability:
* **Key Metrics:** **ROC-AUC: 0.91** and **Macro F1-Score: 0.61** (a more honest metric than the biased 96% Accuracy).
* **Conclusion:** The model is operationally conservative at the default threshold (**Recall: 0.18** / **Precision: 0.38**). The strong contrast between the high AUC and low Recall indicates that while the model correctly ranks risky companies, the decision threshold must be tuned downward to capture the majority of bankruptcy cases.

### Clustering Insights
The unsupervised K-Means model ($K=6$) successfully isolated risk without ever seeing the target label:
* **Cluster 3 ("High Risk"):** Identified a group with a **75% bankruptcy rate**. Characterized by high debt and low efficiency.
* **Cluster 0 & 2 ("Healthy"):** Identified groups with a **0% bankruptcy rate**.
* **Impact:** Proves that financial distress creates a distinct, mathematically observable footprint even without labeled data.

---

## Future Improvements
* **Probability Scoring:** Use the model's probability outputs to price **Credit Default Swaps (CDS)**.
* **Advanced Models:** Test ensemble methods like **Random Forest** or **XGBoost** to potentially improve Recall.
* **Resampling:** Implement **SMOTE** to synthetically oversample the minority class during training.

---

## Libraries Used
* `pandas` & `numpy` (Data Manipulation)
* `matplotlib` & `seaborn` (Visualization)
* `scikit-learn` (Modeling, Scaling, Pipelines, Metrics)
