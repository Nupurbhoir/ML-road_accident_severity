# Project Report: Road Accident Severity Prediction Using Machine Learning

**Course / Subject:** Machine Learning  
**Case Study:** 20 - Road Accident Severity Prediction  

---

## 1. Title
**Road Accident Severity Prediction Using Machine Learning**

---

## 2. Problem Statement
A transport safety authority wants to predict the severity of a road accident (categorised as slight, serious, or fatal) using attributes such as weather conditions, light conditions, road surface, speed limit, vehicle type, and time of day. 

**Dataset Generation:** 
Because the target dataset (Road Traffic Accident Dataset of Addis Ababa City) was unavailable for a reliable deployment, a highly realistic synthetic dataset was generated using `numpy` and `pandas` in Python. The generation algorithm artificially enforces real-world correlations (e.g., higher probabilities of 'Fatal' outcomes when Speed Limit = 70mph and Lighting = 'Darkness - no lighting'). Missing values were intentionally injected into the weather and road condition fields to simulate real-world data collection imperfections and fulfill the problem objectives. The dataset inherently suffers from class imbalance (80% Slight, 15% Serious, 5% Fatal).

---

## 3. Objectives Fulfilled
1. **Exploratory Data Analysis (EDA):** Visualized distributions and correlations within the synthetic data (integrated into the deployed Streamlit Dashboard).
2. **Handle Missing Values:** Implemented `SimpleImputer(strategy='most_frequent')` to handle missing weather and road condition fields.
3. **Encode Categorical Attributes:** Implemented `OneHotEncoder` for fields like Weather, Lighting, and Vehicle Type.
4. **Study Class Imbalance:** Documented that standard models naturally over-predict the 'Slight' class. Mitigated using **SMOTE (Synthetic Minority Over-sampling Technique)**.
5. **Develop and Train Models:** Evaluated Logistic Regression, KNN, Decision Tree, Random Forest, and Gradient Boosting.
6. **Compare Model Performance:** Used **5-Fold Cross-Validation** to extract robust metrics (Accuracy, Precision, Recall, F1-Score).
7. **Select Best-Performing Model:** Selected Logistic Regression (with SMOTE) as it maximized **Recall for the Fatal class**.
8. **Deployment:** Developed an interactive, web-based prediction application using Streamlit.

---

## 4. Comparative Study & 5-Fold Cross-Validation Results
To rigorously evaluate the algorithms without data leakage, a 5-Fold Stratified Cross-Validation was applied alongside SMOTE in an automated Pipeline. 

*Key insight: Recall for Fatal and Serious categories is prioritized over overall Accuracy, as missing a high-severity prediction can lead to inadequate safety interventions.*

| Algorithm | Mean CV Accuracy | Mean Fatal Recall | Mean Serious Recall |
| :--- | :---: | :---: | :---: |
| **Logistic Regression + SMOTE** | **77.75%** | **79.00%** | **65.19%** |
| Gradient Boosting | 83.55% | 74.00% | 52.56% |
| K-Nearest Neighbors (KNN) | 77.45% | 70.00% | 52.56% |
| Random Forest | 83.50% | 62.00% | 39.59% |
| Decision Tree | 81.65% | 58.00% | 40.27% |

**Conclusion:** Logistic Regression provided the most mathematically reliable boundary to maximize the detection of rare, fatal edge-cases.

---

## 5. Deployment
The model is deployed as a fully interactive Web Application using **Streamlit**. 
- The user inputs accident circumstances via dropdowns and sliders.
- The application outputs the **Predicted Severity**, a **Class Probability Matrix**, and automated **Safety Intervention Recommendations** based on the predicted risk.

---

## 6. Final Analysis & Questions Answered

**Q1: Can accident severity be predicted from recorded conditions?**
Yes, environmental factors provide a strong probabilistic indicator of accident severity, though they cannot account for human error or vehicle safety ratings perfectly.

**Q2: Which conditions are most associated with severe outcomes?**
High speed limits (60-70 mph), lack of illumination ('Darkness - no lighting'), and adverse surface conditions (Ice/Snow) show the strongest mathematical correlation to 'Fatal' outcomes.

**Q3: How does class imbalance affect prediction of fatal accidents?**
Without techniques like SMOTE, the model becomes heavily biased toward predicting the majority class ('Slight'), resulting in near-zero Recall for Fatal accidents (high False Negatives).

**Q4: Which algorithm achieves the best recall on rare classes?**
Through our 5-Fold CV testing, Logistic Regression paired with SMOTE achieved the best recall (79%) for the 'Fatal' class. 

**Q5: Which severity categories are most often confused?**
'Slight' and 'Serious' are frequently confused by the model because the boundary separating them often involves data we do not have (e.g., whether the occupants were wearing seatbelts).

**Q6: How well does the model perform on data from a different region?**
Models are highly region-specific. This model would perform poorly in a different city unless retraining via Transfer Learning is applied, as local driving habits, vehicle distributions, and road quality differ drastically globally.

**Q7: Can the model be deployed to guide road safety planning?**
Yes. Transport authorities can input proposed road setups (e.g., a new 50mph road without lights) into the Streamlit app. If the AI flags a high risk of fatal outcomes, planners can intervene early by installing lighting or lowering the speed limit.

**Q8: Limitations of predicting severity from recorded circumstances alone?**
The model lacks data on driver state (intoxication, fatigue, reaction time), precise angle of collision, and the structural integrity of the specific vehicles involved. Therefore, it outputs an environmental *risk probability*, not absolute certainty.
