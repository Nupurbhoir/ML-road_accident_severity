import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

def load_and_preprocess_data(filepath='data/road_accidents.csv'):
    df = pd.read_csv(filepath)
    
    X = df.drop('Accident_Severity', axis=1)
    y = df['Accident_Severity']
    
    # Define categorical and numerical features
    categorical_features = ['Weather_Conditions', 'Light_Conditions', 'Road_Surface_Conditions', 'Vehicle_Type', 'Time_of_Day']
    numerical_features = ['Speed_Limit']
    
    # Preprocessing pipelines
    numeric_transformer = ImbPipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Handle missing values with 'most_frequent' strategy and then one-hot encode
    categorical_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return X, y, preprocessor

def train_and_evaluate():
    print("Loading data...")
    X, y, preprocessor = load_and_preprocess_data()
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Define custom scorers for cross-validation
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        'recall_fatal': make_scorer(recall_score, average=None, labels=['Fatal'])[0],
        'recall_serious': make_scorer(recall_score, average=None, labels=['Serious'])[0]
    }
    
    # 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_model_name = None
    best_recall_fatal = -1
    best_pipeline = None
    
    print("\nTraining models and evaluating using 5-Fold Cross-Validation...")
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Pipeline with SMOTE
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        # Perform K-Fold CV
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Calculate mean scores across the 5 folds
        acc = np.mean(cv_results['test_accuracy'])
        prec = np.mean(cv_results['test_precision_macro'])
        rec = np.mean(cv_results['test_recall_macro'])
        f1 = np.mean(cv_results['test_f1_macro'])
        fatal_recall = np.mean(cv_results['test_recall_fatal'])
        serious_recall = np.mean(cv_results['test_recall_serious'])
        
        print(f"Mean Accuracy: {acc:.4f}")
        print(f"Mean Macro F1: {f1:.4f}")
        print(f"Mean Recall (Fatal): {fatal_recall:.4f}")
        print(f"Mean Recall (Serious): {serious_recall:.4f}")
        
        if fatal_recall > best_recall_fatal:
            best_recall_fatal = fatal_recall
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n==========================================")
    print(f"Best model selected: {best_model_name}")
    print(f"Highest 5-Fold Fatal Recall: {best_recall_fatal:.4f}")
    print(f"==========================================")
    
    # Train the best model on the FULL dataset for final deployment
    print(f"Training {best_model_name} on the full dataset for deployment...")
    best_pipeline.fit(X, y)
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipeline, f'models/best_model.pkl')
    print("Best model saved to models/best_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
