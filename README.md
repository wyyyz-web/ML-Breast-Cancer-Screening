# 🩺 Machine Learning-Based Breast Cancer Diagnosis

This project implements a machine learning framework for binary classification of breast tumors (**benign vs. malignant**) using the Wisconsin Diagnostic Breast Cancer (**WDBC**) dataset. The study proposes an innovative **two-tier diagnostic framework**, combining **high-recall screening** with **high-precision verification** to support reliable clinical decision-making.

---

## 👥 Project Team

**Lead Developer (ML Models & Explainability):** Wang Yuanzhu

**Team Members:** Xin Yongyi, Xu Haowang, Gai Yuxuan, Yun Jiaqi

---

## 🚀 Key Contributions (Wang Yuanzhu)

As the core developer for the modeling phase, my contributions included:

### 🔹 Ensemble Modeling
Implemented state-of-the-art **Random Forest (Bagging)** and **XGBoost (Boosting)** ensemble learning models to capture complex non-linear decision boundaries in breast cancer diagnosis.

### 🔹 Model Optimization
Applied **RandomizedSearchCV** with **5-fold cross-validation** and structural constraints to optimize model performance while minimizing overfitting and improving generalization.

### 🔹 Explainable AI (XAI)
Conducted comprehensive **SHAP (SHapley Additive exPlanations)** analysis to interpret model behavior and identify the most influential clinical features contributing to malignant predictions.

---

# 📖 1. Project Overview

Breast cancer remains one of the most common cancers worldwide. Early and accurate diagnosis is critical for improving treatment outcomes and reducing mortality.

This project addresses the binary classification task of predicting tumor diagnosis based on **569 patient records** and **30 continuous numerical features** extracted from digitized fine-needle aspirate (FNA) images of breast masses.

Dataset:

- Wisconsin Diagnostic Breast Cancer (WDBC)
- 569 samples
- 30 numerical features
- Binary labels:
  - Benign (B)
  - Malignant (M)

---

## 🏥 The Two-Tier Diagnostic Framework

To address the clinical trade-off between sensitivity and specificity, the project proposes a hierarchical diagnostic strategy:

### Tier 1 — Initial Screening

**Logistic Regression (Threshold = 0.35)**

Goal:

- Maximize sensitivity
- Minimize missed malignant cases
- Act as a first-pass screening system

Performance:

- Recall = **97.62%**

This stage prioritizes patient safety by reducing false negatives.

---

### Tier 2 — Expert Verification

**Constrained Random Forest**

Goal:

- Provide highly reliable malignant predictions
- Reduce false positives
- Assist clinicians with high-confidence diagnosis

Performance:

- Precision = **100%**

This stage acts as an expert verification layer for suspicious cases identified during screening.

---

# ⚙️ 2. Methodology & Pipeline

The entire machine learning pipeline was designed to ensure reproducibility and eliminate data leakage.

## Workflow

```text
Raw WDBC Dataset
        │
        ▼
Train/Test Split (80/20)
        │
        ▼
StandardScaler
(Train Fit Only)
        │
        ▼
SMOTE Oversampling
(Training Data Only)
        │
        ▼
Model Training
 ├── Logistic Regression
 ├── Random Forest
 └── XGBoost
        │
        ▼
Evaluation
        │
        ▼
SHAP Explainability
```

---

## Data Preparation

- Stratified Train/Test Split
- 80% Training Set
- 20% Test Set
- Random Seed = 42

Benefits:

- Preserves class distribution
- Ensures reproducibility

---

## Data Preprocessing

### Standardization

Feature scaling was performed using:

```python
StandardScaler()
```

Important:

- Fitted only on training data
- Applied to test data afterward

This prevents information leakage.

---

### Class Balancing

To address class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied exclusively to the training set.

Benefits:

- Improved minority class representation
- Enhanced model robustness
- Reduced bias toward majority class

---

## Models Implemented

### 1. Logistic Regression (L2 Regularization)

Purpose:

- Baseline classifier
- High-sensitivity screening

Characteristics:

- Interpretable
- Fast
- Robust

---

### 2. Random Forest (Constrained)

Purpose:

- High-confidence verification

Characteristics:

- Ensemble Bagging Method
- Reduced variance
- Strong robustness against overfitting

Optimization:

- RandomizedSearchCV
- Structural constraints
- 5-fold cross-validation

---

### 3. XGBoost

Purpose:

- Advanced Boosting Benchmark

Characteristics:

- Gradient Boosting Trees
- Strong predictive performance
- Handles complex feature interactions

---

# 📊 3. Results Summary

The **Constrained Random Forest** achieved **perfect precision (1.0000)** on the held-out test set, effectively eliminating false-positive malignant predictions.

## Test Set Performance

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---------|---------|---------|---------|---------|
| L2_SMOTE (LR) | 0.9737 | 0.9756 | 0.9524 | 0.9977 |
| Constrained RF | 0.9737 | 1.0000 | 0.9286 | 0.9967 |
| XGBoost | 0.9649 | 1.0000 | 0.9048 | 0.9940 |

> Performance metrics are reported on the held-out test set (n = 114).

---

## Key Findings

### Logistic Regression

Strengths:

- Highest Recall
- Suitable for screening

Recall:

```text
97.62%
```

---

### Constrained Random Forest

Strengths:

- Perfect Precision
- Excellent ROC-AUC
- Suitable for expert verification

Precision:

```text
100%
```

---

### XGBoost

Strengths:

- Competitive performance
- Strong generalization

Precision:

```text
100%
```

---

# 📈 4. Explainability (SHAP Analysis)

Machine learning models often function as "black boxes." To improve transparency and trustworthiness, SHAP analysis was employed.

## Why SHAP?

SHAP provides:

- Global feature importance
- Local prediction explanations
- Consistent interpretation framework

---

## Top Predictive Features

The SHAP results identified the following features as the strongest indicators of malignancy:

1. `area_worst`
2. `perimeter_worst`
3. `concave_points_worst`

These findings align closely with established cytopathological knowledge and clinical observations.

---

## Clinical Interpretation

The analysis demonstrates that:

- Larger tumor area
- Larger tumor perimeter
- Greater concavity-related abnormalities

are strongly associated with malignant diagnoses.

This consistency enhances confidence in the model's medical relevance.

---

# 🛠️ Technologies Used

## Programming Language

- Python 3.x

## Machine Learning

- Scikit-learn
- XGBoost
- imbalanced-learn

## Explainable AI

- SHAP

## Data Processing

- NumPy
- Pandas

## Visualization

- Matplotlib
- Seaborn

---

# 📂 Project Structure

```text
Machine-Learning-Based-Breast-Cancer-Diagnosis/
│
├── data/
│   └── WDBC dataset
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── modeling.ipynb
│   └── shap_analysis.ipynb
│
├── figures/
│   ├── ROC_curves.png
│   ├── PR_curves.png
│   └── shap_summary.png
│
├── report/
│   └── Technical_Report.pdf
│
├── requirements.txt
│
└── README.md
```

---

# 📚 References

### Dataset

Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995).

*Machine Learning Techniques to Diagnose Breast Cancer.*

---

### Class Imbalance

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).

*SMOTE: Synthetic Minority Over-sampling Technique.*

Journal of Artificial Intelligence Research.

---

### Explainable AI

Lundberg, S. M., & Lee, S.-I. (2017).

*A Unified Approach to Interpreting Model Predictions.*

Advances in Neural Information Processing Systems (NeurIPS).

---

# ⚠️ Statement on AI Usage

AI tools including **ChatGPT** and **Doubao** were utilized for assistance with documentation refinement, language polishing, and report synthesis.

All machine learning implementation, experimental design, model development, parameter optimization, evaluation, and result interpretation were independently completed by the project team.

---

# 📄 Technical Report

Full methodological details, experimental settings, optimization procedures, and discussion of findings are available in the accompanying technical report.

---

## ⭐ Highlights

- Two-Tier Clinical Diagnostic Framework
- Logistic Regression + Random Forest + XGBoost
- SMOTE-Based Class Balancing
- Hyperparameter Optimization via RandomizedSearchCV
- SHAP Explainability Analysis
- 100% Precision Random Forest Verification Model
- Reproducible End-to-End Machine Learning Pipeline
