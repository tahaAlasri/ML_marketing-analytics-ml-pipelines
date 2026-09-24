# 📊 Marketing Analytics & Customer Propensity Modeling
### *Production-Grade Supervised Machine Learning Pipelines for Customer Behavior & Ad Conversion*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Overview

This repository houses two comprehensive, end-to-end Machine Learning case studies focused on **Customer Conversion, Propensity Scoring, and Marketing Analytics**. 

Rather than basic exploratory scripts, these analyses are engineered using **industry-grade machine learning pipelines**. They strictly adhere to modern MLOps and statistical standards—including **leakage-free stratified splits**, **domain-driven feature engineering**, **skewness-aware data transformations**, and **stratified cross-validation**.

---

## 🏗️ Core Engineering Highlights

* 🛡️ **Zero Data Leakage:** All data preprocessing (imputation, scaling, one-hot encoding, skewness transformation) and feature engineering are fitted **strictly on the training splits** and encapsulated inside `scikit-learn` pipelines.
* ⚙️ **Modular Preprocessing Pipelines (`ColumnTransformer`):**
  * **Log Pipeline:** `FunctionTransformer(np.log1p)` coupled with `RobustScaler` for highly skewed financial and behavioral metrics.
  * **Numeric Pipeline:** Median imputation with `RobustScaler` to insulate against outliers.
  * **Categorical Pipeline:** Missing category imputation with `OneHotEncoder(handle_unknown='ignore')`.
* ⚖️ **Imbalanced Class Optimization:** Prioritizes **ROC-AUC**, **F1-Score**, and **Recall** with `class_weight='balanced'` to prevent bias toward non-converting majorities.
* 🔬 **Rigorous Model Validation:** Employs **5-Fold Stratified Cross-Validation (`StratifiedKFold`)** with mean and standard deviation reporting, followed by out-of-sample holdout test verification.

---

## 📂 Repository Structure

```text
├── data/
│   ├── marketing_campaign.csv          # Customer personality & campaign response dataset (2,240 rows)
│   └── Social_Network_Ads.csv          # Targeted social ads demographic conversion dataset (400 rows)
├── notebooks/
│   ├── Marketing_Campaign_Analysis.ipynb  # End-to-end propensity modeling pipeline
│   └── Social_Network_Ads_Analysis.ipynb  # Demographic ad conversion & classification benchmark
├── .gitignore                          # Clean git tracking exclusions
├── requirements.txt                    # Reproducible environment dependencies
└── README.md                           # Documentation & benchmarks
```

---

## 🔬 Case Study 1: Marketing Campaign Propensity Modeling

* **Notebook:** [`Marketing_Campaign_Analysis.ipynb`](notebooks/Marketing_Campaign_Analysis.ipynb)
* **Goal:** Predict whether a customer will accept a promotional campaign (`Response`), enabling targeted marketing with minimized ad fatigue and maximized ROI.
* **Key Feature Engineering:**
  * `Customer_Tenure_Days`: Recency from customer enrollment to baseline reference date.
  * `Total_Spend`: Aggregate customer expenditure across 6 product categories (Wine, Meat, Fish, Gold, Sweets, Fruits).
  * `Total_Purchases`: Omnichannel purchase intensity across Web, Catalog, Store, and Discount Deals.
  * `Total_Accepted_Campaigns`: Loyalty and historical engagement propensity score.
  * `Is_Parent` & `Total_Children`: Household demographic segmentation.

### 🏆 Benchmark Results

| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test Recall | Test Precision | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Propensity Baseline)** | **0.9096 (±0.020)** | **0.9071** | 84.15% | **76.12%** | 48.11% | **58.96%** |
| **Gradient Boosting Classifier (Boosting)** | 0.8956 (±0.020) | **0.9022** | **89.96%** | 40.30% | **84.38%** | 54.55% |
| **Random Forest Classifier (Bagging)** | 0.8804 (±0.030) | 0.8678 | 81.47% | 67.16% | 42.45% | 52.02% |

> **Strategic Takeaway:** Logistic Regression configured with balanced class weights provides the highest campaign sensitivity (Recall: **76.12%**, ROC-AUC: **0.9071**), ensuring high capture of potential buyers for proactive marketing campaigns.

---

## 🎯 Case Study 2: Social Network Ad Conversion Classification

* **Notebook:** [`Social_Network_Ads_Analysis.ipynb`](notebooks/Social_Network_Ads_Analysis.ipynb)
* **Goal:** Classify user purchase decisions based on age and estimated income to optimize ad spend allocation.
* **Key Feature Engineering:**
  * `Salary_Per_Age`: Discretionary spending capability proxy.
  * `Is_Senior` & `High_Salary`: Non-linear decision boundary indicators.
  * `Salary_Age_Product`: Interaction term capturing wealth accumulation velocity.

### 🏆 Benchmark Results

| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test Recall | Test Precision | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier** | 0.9546 (±0.010) | **0.9523** | 88.75% | 86.21% | 83.33% | 84.75% |
| **Random Forest Classifier** | **0.9581 (±0.016)** | 0.9506 | 88.75% | 86.21% | 83.33% | 84.75% |
| **Logistic Regression (Parametric)** | 0.9572 (±0.008) | 0.9500 | **90.00%** | **89.66%** | **83.87%** | **86.67%** |
| **Support Vector Machine (RBF Kernel)** | 0.9469 (±0.025) | 0.9344 | 88.75% | 89.66% | 81.25% | 85.25% |

> **Strategic Takeaway:** All tuned classification pipelines achieve a Test ROC-AUC exceeding **0.93**, with Logistic Regression and Gradient Boosting delivering high accuracy (**90.0%**) and high recall (**89.66%**).

---

## 🚀 Quickstart & Setup

### 1. Clone Repository
```bash
git clone https://github.com/tahaAlasri/ML_marketing-analytics-ml-pipelines.git
cd ML_marketing-analytics-ml-pipelines
```

### 2. Create and Activate Virtual Environment
```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebooks
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and run any of the notebooks. Datasets will resolve and load automatically.

---

## 📄 License
This project is licensed under the MIT License - feel free to use and adapt it for learning and portfolio purposes.
