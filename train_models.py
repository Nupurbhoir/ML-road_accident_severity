import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    make_scorer, recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def load_and_preprocess_data(filepath='data/road_accidents.csv'):
    df = pd.read_csv(filepath)
    
    X = df.drop('Accident_Severity', axis=1)
    y = df['Accident_Severity']
    
    categorical_features = ['Weather_Conditions', 'Light_Conditions', 'Road_Surface_Conditions', 'Vehicle_Type', 'Time_of_Day']
    numerical_features = ['Speed_Limit']
    
    numeric_transformer = ImbPipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return df, X, y, preprocessor

def generate_eda_charts(df):
    os.makedirs('assets', exist_ok=True)
    sns.set_theme(style="whitegrid")
    palette = {'Slight': '#10b981', 'Serious': '#f59e0b', 'Fatal': '#ef4444'}
    
    # 1. Missing Values Before Preprocessing
    plt.figure(figsize=(8, 4.5))
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    ax = sns.barplot(x=missing.index, y=missing.values, palette='viridis')
    plt.title('Missing Values per Feature Before Preprocessing', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Missing Count', fontsize=12)
    plt.xlabel('Features', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig('assets/missing_values.png', dpi=300)
    plt.close()

    # 2. Severity Distribution
    plt.figure(figsize=(8, 4.5))
    sev_counts = df['Accident_Severity'].value_counts()
    ax = sns.barplot(x=sev_counts.index, y=sev_counts.values, palette=palette)
    plt.title('Accident Severity Class Distribution (Severe Class Imbalance)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Count of Accidents', fontsize=12)
    plt.xlabel('Severity Level', fontsize=12)
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{100 * height / total:.1f}%'
        ax.annotate(f'{int(height)}\n({percentage})', (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig('assets/severity_distribution.png', dpi=300)
    plt.close()

    # 3. Severity vs Light Conditions
    plt.figure(figsize=(10, 5.5))
    light_sev = pd.crosstab(df['Light_Conditions'], df['Accident_Severity'], normalize='index') * 100
    light_sev = light_sev[['Slight', 'Serious', 'Fatal']]
    ax = light_sev.plot(kind='bar', stacked=True, color=['#10b981', '#f59e0b', '#ef4444'], figsize=(10, 5.5))
    plt.title('Accident Severity Proportions by Light Conditions', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xlabel('Light Conditions', fontsize=12)
    plt.legend(title='Severity', loc='upper right')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('assets/severity_vs_light.png', dpi=300)
    plt.close()

    # 4. Severity vs Speed Limit
    plt.figure(figsize=(9, 5))
    ax = sns.countplot(data=df, x='Speed_Limit', hue='Accident_Severity', palette=palette)
    plt.title('Accident Counts Across Speed Limits by Severity', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Accident Count', fontsize=12)
    plt.xlabel('Speed Limit (mph)', fontsize=12)
    plt.legend(title='Severity')
    plt.tight_layout()
    plt.savefig('assets/severity_vs_speed.png', dpi=300)
    plt.close()

    # 5. Severity vs Road Surface Conditions
    plt.figure(figsize=(10, 5.5))
    road_sev = pd.crosstab(df['Road_Surface_Conditions'], df['Accident_Severity'], normalize='index') * 100
    road_sev = road_sev[['Slight', 'Serious', 'Fatal']]
    ax = road_sev.plot(kind='bar', stacked=True, color=['#10b981', '#f59e0b', '#ef4444'], figsize=(10, 5.5))
    plt.title('Accident Severity Proportions by Road Surface Conditions', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xlabel('Road Surface Conditions', fontsize=12)
    plt.legend(title='Severity', loc='upper right')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('assets/severity_vs_road_surface.png', dpi=300)
    plt.close()

def fatal_recall_fn(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=['Fatal'], average=None, zero_division=0)[0]

def serious_recall_fn(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=['Serious'], average=None, zero_division=0)[0]

def train_and_evaluate():
    print("Loading data...")
    df, X, y, preprocessor = load_and_preprocess_data()
    
    print("Generating EDA charts...")
    generate_eda_charts(df)
    
    models = {
        'Logistic Regression + SMOTE': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        'recall_fatal': make_scorer(fatal_recall_fn),
        'recall_serious': make_scorer(serious_recall_fn)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    best_model_name = None
    best_recall_fatal = -1
    best_pipeline = None
    
    print("\nTraining models and evaluating using 5-Fold Cross-Validation...")
    
    for name, model in models.items():
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        acc = np.mean(cv_results['test_accuracy'])
        prec = np.mean(cv_results['test_precision_macro'])
        rec = np.mean(cv_results['test_recall_macro'])
        f1 = np.mean(cv_results['test_f1_macro'])
        fatal_rec = np.mean(cv_results['test_recall_fatal'])
        serious_rec = np.mean(cv_results['test_recall_serious'])
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (Macro)': prec,
            'Recall (Macro)': rec,
            'F1-score (Macro)': f1,
            'Fatal Recall': fatal_rec,
            'Serious Recall': serious_rec
        })
        
        if fatal_rec > best_recall_fatal:
            best_recall_fatal = fatal_rec
            best_model_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results)
    print("\n=================== 5-FOLD CV RESULTS SUMMARY ===================")
    print(results_df.to_string(index=False))
    print("=================================================================\n")
    
    # Generate Model Comparison Chart
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-score (Macro)', 'Fatal Recall']
    df_plot = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    
    ax = sns.barplot(data=df_plot, x='Model', y='Score', hue='Metric', palette='Set2')
    plt.title('5-Fold Cross-Validation Model Comparison Across All Metrics', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 1.05)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.legend(title='Evaluation Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('assets/model_comparison.png', dpi=300)
    plt.close()
    
    # Train Best Model & Generate Confusion Matrix
    print(f"Training best model ({best_model_name}) on Train-Test split for Confusion Matrix & Deployment...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    
    labels = ['Slight', 'Serious', 'Fatal']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    print("\nConfusion Matrix (Holdout Test Set 20%):")
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Predicted {l}" for l in labels])
    print(cm_df)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(f'Confusion Matrix: {best_model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Severity', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Severity', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix.png', dpi=300)
    plt.close()
    
    # Re-fit on full dataset for Streamlit deployment
    print(f"\nRefitting {best_model_name} on full dataset (10,000 records) for deployment...")
    best_pipeline.fit(X, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipeline, 'models/best_model.pkl')
    print("Best model saved to models/best_model.pkl")

    return results_df, cm_df

if __name__ == "__main__":
    train_and_evaluate()
