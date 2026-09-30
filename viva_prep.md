# Road Accident Severity Prediction - Viva Defense Guide

This guide contains concise responses and key technical metrics for project presentation.

---

## 1. 🎤 Project Overview Script
> *"My project is **Nexus AI**, a predictive machine learning system designed to estimate road traffic accident severity (Slight, Serious, or Fatal) based on environmental telemetry—such as speed limits, lighting, weather, and road surface conditions."*
>
> *"To handle class imbalance (80.34% slight, 14.66% serious, 5.00% fatal), I developed a pipeline incorporating **SMOTE oversampling** and **5-Fold Stratified Cross-Validation**. The selected model—**Logistic Regression + SMOTE**—achieves a **78.80% Fatal Recall** and **73.14% Macro Recall**, reducing Fatal False Negatives to 2.0% on the holdout test set. The model is deployed as an interactive Streamlit application."*

---

## 2. 📊 5-Fold Stratified Cross-Validation Performance Metrics

| Algorithm | Mean CV Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Fatal Recall | Serious Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression + SMOTE** | **77.23%** | **60.81%** | **73.14%** | **64.72%** | **78.80%** | **60.44%** |
| Gradient Boosting | 83.19% | 65.77% | 71.28% | 68.17% | 69.80% | 54.84% |
| K-Nearest Neighbors | 77.03% | 56.76% | 66.12% | 60.03% | 66.40% | 49.18% |
| Random Forest | 83.18% | 65.82% | 64.63% | 65.17% | 60.20% | 41.47% |
| Decision Tree | 81.25% | 61.37% | 62.41% | 61.83% | 57.80% | 39.02% |

---

## 3. 🧩 Holdout Confusion Matrix Summary

```
                 Predicted Slight  Predicted Serious  Predicted Fatal
Actual Slight               1285                306               16
Actual Serious                53                191               49
Actual Fatal                   2                 19               79
```

- **Slight (Actual 1607):** 1,285 correctly predicted (80.0%).
- **Serious (Actual 293):** 191 correctly predicted (65.2%).
- **Fatal (Actual 100):** 79 correctly predicted (79.0%).
- **Fatal False Negatives:** 2 out of 100 Fatal cases (2.0%) misclassified as Slight.

---

## 4. 🧠 Key Questions & Technical Answers

### Q1: Why Logistic Regression + SMOTE over Random Forest or Gradient Boosting?
**Answer:** *"While Gradient Boosting achieved higher overall accuracy (83.19%), it yielded a lower Fatal Recall (69.80%). In safety-critical applications, minimizing False Negatives for high-severity cases is prioritized. Logistic Regression + SMOTE achieved the highest **Fatal Recall (78.80%)** and **Macro Recall (73.14%)**."*

### Q2: How did you handle missing values?
**Answer:** *"Missing values (5% in weather and road surface attributes) were imputed using `SimpleImputer(strategy='most_frequent')` inside an `ImbPipeline` to ensure preprocessing occurred independently within each cross-validation fold."*

### Q3: What is SMOTE?
**Answer:** *"SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic minority class samples along line segments connecting nearest neighbors in feature space, balancing class proportions during training without duplicating rows."*

### Q4: How does the model perform on data from a different region?
**Answer:** *"Cross-region performance was not directly evaluated because the project uses a synthetic dataset. Deployment to another region would require validation and retraining using representative local accident data from that jurisdiction."*

### Q5: How is the selected model displayed in the app?
**Answer:** *"When a prediction is initialized, the Streamlit app prominently displays `Selected Model: Logistic Regression + SMOTE` alongside validation metric summaries and confidence probability charts."*
