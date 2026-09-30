# Road Accident Severity Prediction - The Ultimate Viva Guide (20-Mark Scorecard)

This guide contains your exact speech scripts, technical definitions, and answers to guarantee full marks in your project evaluation.

---

## 1. 🎤 The 45-Second Project Introduction (Say This First!)
> *"Good morning/afternoon Examiner. My project is **Nexus AI**, an intelligent Road Accident Severity Predictor. Using machine learning, it analyzes environmental telemetry—such as speed limit, lighting, weather, and road surface conditions—to predict collision severity as Slight, Serious, or Fatal."*
>
> *"Because real-world traffic data suffers from extreme class imbalance (80% slight and only 5% fatal), standard models naturally fail. I built a pipeline incorporating **SMOTE oversampling** and **5-Fold Stratified Cross-Validation**. My final selected model—**Logistic Regression + SMOTE**—achieves a **78.80% Recall on Fatal accidents** with a minimal **2.0% False Negative rate**, deployed live on a Streamlit Web Application."*

---

## 2. 📊 Verified 5-Fold Cross-Validation Metrics Table

| Algorithm | Mean CV Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Fatal Recall | Serious Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression + SMOTE** | **77.23%** | **60.81%** | **73.14%** | **64.72%** | **78.80%** | **60.44%** |
| Gradient Boosting | 83.19% | 65.77% | 71.28% | 68.17% | 69.80% | 54.84% |
| K-Nearest Neighbors | 77.03% | 56.76% | 66.12% | 60.03% | 66.40% | 49.18% |
| Random Forest | 83.18% | 65.82% | 64.63% | 65.17% | 60.20% | 41.47% |
| Decision Tree | 81.25% | 61.37% | 62.41% | 61.83% | 57.80% | 39.02% |

---

## 3. 🧩 Confusion Matrix Explanation (Holdout Test Set)

```
                 Predicted Slight  Predicted Serious  Predicted Fatal
Actual Slight               1285                306               16
Actual Serious                53                191               49
Actual Fatal                   2                 19               79
```

- **Slight (Actual 1607):** 1,285 correctly predicted (80.0%).
- **Serious (Actual 293):** 191 correctly predicted (65.2%).
- **Fatal (Actual 100):** 79 correctly predicted (79.0%).
- **Key Safety Victory:** Only **2 out of 100 Fatal cases** were misclassified as Slight (2% False Negative Rate).

---

## 4. 🧠 Top Viva Questions & Instant Master Answers

### Q1: Why Logistic Regression + SMOTE over Random Forest or Gradient Boosting?
**Answer:** *"While Gradient Boosting gives higher raw accuracy (83.19%), it misses over 30% of fatal collisions because it biases toward the majority class. In safety engineering, missing a fatal accident is a risk to human life. Logistic Regression + SMOTE yielded the highest **Fatal Recall (78.80%)** and **Macro Recall (73.14%)**, minimizing False Negatives."*

### Q2: How did you handle missing values?
**Answer:** *"Missing values were injected into 5% of weather and road surface fields. I used `SimpleImputer(strategy='most_frequent')` inside an `ImbPipeline`. This ensured that imputation occurred inside each cross-validation fold independently, preventing data leakage."*

### Q3: What is SMOTE and how does it work?
**Answer:** *"SMOTE stands for Synthetic Minority Over-sampling Technique. Instead of simply duplicating minority class rows, SMOTE selects minority samples, computes their k-nearest neighbors in feature space, and generates synthetic new samples along the line segments connecting them."*

### Q4: Do you need a separate backend for Streamlit?
**Answer:** *"No. Streamlit acts as both the frontend renderer and the Python backend server. The trained model binary (`best_model.pkl`) is loaded into memory directly, executing real-time vector inference on `localhost:8501` without API overhead."*

### Q5: How is your model choice displayed in the UI?
**Answer:** *"In the Streamlit app, when the user clicks 'INITIALIZE THREAT ANALYSIS', the result section explicitly displays a callout badge: `Selected Model: Logistic Regression + SMOTE` along with validation metric summaries and an interactive confidence donut chart."*
