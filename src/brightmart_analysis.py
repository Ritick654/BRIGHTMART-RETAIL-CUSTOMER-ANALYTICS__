"""
BrightMart Retail: End-to-End Analytical Modelling Pipeline
===========================================================
Tasks:
- Task A: Data Preparation and Exploratory Analysis
- Task B: Linear Regression — Sales Prediction
- Task C: Decision Tree — Customer Classification
- Task D: K-Means — Customer Segmentation
- Task E: Random Forest — Campaign Response Prediction
- Task F: Model Comparison & Artifact Generation

Author: Data Analytics & ML Team
"""

import sys
import os

# Handle Windows UTF-8 stdout encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Set local matplotlib cache directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['MPLCONFIGDIR'] = os.path.join(BASE_DIR, "outputs", ".mplconfig")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = True

# Paths
DATA_RAW = os.path.join(BASE_DIR, "data", "BrightMart_ML_Dataset.csv")
DATA_CLEAN = os.path.join(BASE_DIR, "data", "BrightMart_Cleaned_Dataset.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "evaluation_metrics.json")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def run_task_a():
    print("=" * 70)
    print("TASK A: DATA PREPARATION AND EXPLORATORY ANALYSIS")
    print("=" * 70)
    
    # 1. Load dataset
    df_raw = pd.read_csv(DATA_RAW)
    print(f"Loaded raw data shape: {df_raw.shape}")
    print("\nFirst 5 rows of raw data:")
    print(df_raw.head())
    
    print("\nData Types and Non-Null Counts:")
    print(df_raw.info())
    
    print("\nDescriptive Statistics (Raw Data):")
    print(df_raw.describe())
    
    # 2. Check duplicates
    duplicate_count = df_raw.duplicated().sum()
    print(f"\nDuplicate rows detected: {duplicate_count}")
    df_clean = df_raw.drop_duplicates().copy()
    print(f"Shape after removing duplicates: {df_clean.shape}")
    
    # 3. Check missing values
    missing_summary = df_clean.isnull().sum()
    print(f"\nMissing values per column:\n{missing_summary[missing_summary > 0]}")
    
    # Imputation with justification:
    # - Annual_Income: Impute with median (INR thousands) to avoid skewness from high earners
    income_median = df_clean['Annual_Income'].median()
    df_clean['Annual_Income'] = df_clean['Annual_Income'].fillna(income_median)
    
    # - Spending_Score: Impute with median (score 1-100)
    score_median = df_clean['Spending_Score'].median()
    df_clean['Spending_Score'] = df_clean['Spending_Score'].fillna(score_median)
    
    # - Ad_Spend: Impute with median advertising expenditure
    ad_median = df_clean['Ad_Spend'].median()
    df_clean['Ad_Spend'] = df_clean['Ad_Spend'].fillna(ad_median)
    
    print(f"Imputed missing values: Annual_Income median={income_median}, Spending_Score median={score_median}, Ad_Spend median={ad_median}")
    
    # 4. Check unusual or invalid values
    print("\nChecking for invalid/anomalous values:")
    # Age check (must be 18 to 100)
    invalid_age = df_clean[(df_clean['Age'] < 18) | (df_clean['Age'] > 100)]
    print(f"Invalid Age entries (Age < 18 or > 100): {len(invalid_age)}")
    if len(invalid_age) > 0:
        age_median = df_clean.loc[(df_clean['Age'] >= 18) & (df_clean['Age'] <= 100), 'Age'].median()
        df_clean.loc[(df_clean['Age'] < 18) | (df_clean['Age'] > 100), 'Age'] = age_median
        print(f"Corrected invalid age(s) to median age: {age_median}")
        
    # Spending Score check (must be 1 to 100)
    invalid_score = df_clean[(df_clean['Spending_Score'] < 1) | (df_clean['Spending_Score'] > 100)]
    print(f"Invalid Spending_Score entries (<1 or >100): {len(invalid_score)}")
    if len(invalid_score) > 0:
        df_clean['Spending_Score'] = df_clean['Spending_Score'].clip(1, 100)
        print("Clipped invalid spending score entries to valid range [1, 100].")
        
    # Sales check (must be >= 0)
    invalid_sales = df_clean[df_clean['Sales'] < 0]
    print(f"Negative Sales entries: {len(invalid_sales)}")
    if len(invalid_sales) > 0:
        sales_median = df_clean.loc[df_clean['Sales'] >= 0, 'Sales'].median()
        df_clean.loc[df_clean['Sales'] < 0, 'Sales'] = sales_median
        print(f"Corrected negative sales to positive median: {sales_median}")
        
    # Ensure proper data types
    df_clean['Age'] = df_clean['Age'].astype(int)
    df_clean['Spending_Score'] = df_clean['Spending_Score'].astype(int)
    df_clean['Purchase_Frequency'] = df_clean['Purchase_Frequency'].astype(int)
    
    # Save cleaned dataset
    df_clean.to_csv(DATA_CLEAN, index=False)
    print(f"\nCleaned dataset saved successfully to: {DATA_CLEAN}")
    print(f"Final Cleaned Dataset Shape: {df_clean.shape}")
    
    # 5. Exploratory Data Visualisations
    # Plot 1: Sales Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df_clean['Sales'], kde=True, color='#2b5c8f', bins=20)
    plt.title("Distribution of Sales (INR Thousands)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Sales (INR '000)")
    plt.ylabel("Frequency")
    plt.axvline(df_clean['Sales'].mean(), color='red', linestyle='--', label=f"Mean: INR {df_clean['Sales'].mean():.1f}k")
    plt.axvline(df_clean['Sales'].median(), color='orange', linestyle=':', label=f"Median: INR {df_clean['Sales'].median():.1f}k")
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "01_sales_distribution.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    
    # Plot 2: Ad Spend vs Sales Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df_clean, x='Ad_Spend', y='Sales', hue='Customer_Segment', palette='Set2', s=70, alpha=0.85)
    sns.regplot(data=df_clean, x='Ad_Spend', y='Sales', scatter=False, color='#e74c3c', line_kws={'linestyle':'--', 'linewidth':2})
    plt.title("Ad Spend vs. Store Sales (EDA)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Advertising Expenditure (INR '000)")
    plt.ylabel("Sales (INR '000)")
    plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "02_ad_spend_vs_sales_eda.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    
    # Plot 3: Correlation Matrix
    numeric_cols = ['Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency', 'Ad_Spend', 'Sales']
    plt.figure(figsize=(8, 6))
    corr = df_clean[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', cbar=True, square=True)
    plt.title("Feature Correlation Matrix", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    p3 = os.path.join(PLOTS_DIR, "03_correlation_matrix.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    
    # Plot 4: Customer Demographics & Discount Usage
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df_clean, x='Customer_Segment', y='Annual_Income', hue='Customer_Segment', legend=False, ax=axes[0], palette='pastel')
    axes[0].set_title("Annual Income by Customer Segment", fontweight='bold')
    axes[0].set_ylabel("Annual Income (INR '000)")
    
    sns.countplot(data=df_clean, x='Discount_Usage', hue='Campaign_Response', ax=axes[1], palette='Set1')
    axes[1].set_title("Discount Usage vs Campaign Response", fontweight='bold')
    axes[1].set_ylabel("Count of Customers")
    plt.tight_layout()
    p4 = os.path.join(PLOTS_DIR, "04_customer_demographics.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    
    print(f"Task A plots saved to {PLOTS_DIR}")
    return df_clean


def run_task_b(df):
    print("\n" + "=" * 70)
    print("TASK B: LINEAR REGRESSION — SALES PREDICTION")
    print("=" * 70)
    
    X = df[['Ad_Spend']]
    y = df['Sales']
    
    # Train-test split (80-20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    
    # Train model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)
    
    # Evaluation metrics
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    slope = lr_model.coef_[0]
    intercept = lr_model.intercept_
    
    print(f"Regression Equation: Sales = {intercept:.2f} + {slope:.2f} * Ad_Spend")
    print(f"Slope (Coefficient): {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"Mean Absolute Error (MAE): INR {mae_test:.2f} thousand")
    print(f"R-squared Score (R²): {r2_test:.4f}")
    
    # Business scenario prediction
    new_budget = 35.0  # INR 35,000 ad budget
    pred_sales_35k = lr_model.predict([[new_budget]])[0]
    print(f"\nScenario Prediction for new Ad Spend = INR {new_budget:.1f}k:")
    print(f"Predicted Sales = INR {pred_sales_35k:.2f} thousand")
    
    # Save model artifact
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "linear_regression.pkl"))
    
    # Visualisation
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color='#3498db', alpha=0.6, label='Train Actuals')
    plt.scatter(X_test, y_test, color='#2ecc71', alpha=0.9, marker='s', label='Test Actuals')
    
    # Regression line
    x_range = np.linspace(X['Ad_Spend'].min() - 2, X['Ad_Spend'].max() + 2, 100).reshape(-1, 1)
    y_line = lr_model.predict(x_range)
    plt.plot(x_range, y_line, color='#e74c3c', linewidth=2.5, label=f'Fitted Line: Sales = {intercept:.1f} + {slope:.2f}*Ad_Spend')
    
    # Scenario point
    plt.scatter([new_budget], [pred_sales_35k], color='gold', edgecolor='black', s=160, zorder=5, label=f'Scenario: INR {new_budget}k Ad -> INR {pred_sales_35k:.1f}k Sales')
    
    plt.title(f"Linear Regression: Sales vs Advertising Expenditure (R² = {r2_test:.3f})", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Advertising Expenditure (INR '000)")
    plt.ylabel("Store Sales (INR '000)")
    plt.legend(frameon=True)
    plt.tight_layout()
    p_reg = os.path.join(PLOTS_DIR, "05_linear_regression_fit.png")
    plt.savefig(p_reg, dpi=300)
    plt.close()
    
    results = {
        'slope': float(slope),
        'intercept': float(intercept),
        'mae': float(mae_test),
        'r2': float(r2_test),
        'scenario_budget': float(new_budget),
        'scenario_predicted_sales': float(pred_sales_35k)
    }
    return results, (X_train, X_test, y_train, y_test)


def run_task_c(df):
    print("\n" + "=" * 70)
    print("TASK C: DECISION TREE — CUSTOMER CLASSIFICATION")
    print("=" * 70)
    
    # Feature set and target
    # Input features: Age, Annual_Income, Purchase_Frequency, Discount_Usage
    features = ['Age', 'Annual_Income', 'Purchase_Frequency', 'Discount_Usage']
    X = df[features].copy()
    
    # Target: Campaign_Response ('Yes' / 'No' -> 1 / 0)
    y = (df['Campaign_Response'] == 'Yes').astype(int)
    
    # Encode categorical Discount_Usage
    # Ordinal mapping preserves business semantics: Low < Medium < High
    discount_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    X['Discount_Usage_Encoded'] = X['Discount_Usage'].map(discount_mapping)
    feature_names = ['Age', 'Annual_Income', 'Purchase_Frequency', 'Discount_Usage_Encoded']
    X_num = X[feature_names]
    
    # Stratified split to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Classification Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    print(f"Test Class Balance (0=No, 1=Yes): {np.bincount(y_test)}")
    
    # Train Decision Tree (max_depth=3 for interpretability & preventing overfitting)
    dt_model = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Predict
    y_pred = dt_model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nDecision Tree Test Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
    
    # Save model
    joblib.dump(dt_model, os.path.join(MODELS_DIR, "decision_tree.pkl"))
    
    # Visualise Decision Tree
    plt.figure(figsize=(16, 9))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=['No', 'Yes'],
        filled=True,
        rounded=True,
        fontsize=11
    )
    plt.title("Decision Tree Visualisation (Max Depth = 3)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    p_tree = os.path.join(PLOTS_DIR, "06_decision_tree_visualization.png")
    plt.savefig(p_tree, dpi=300)
    plt.close()
    
    # Visualise Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
    plt.title("Decision Tree Confusion Matrix", fontsize=12, fontweight='bold', pad=10)
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    p_cm = os.path.join(PLOTS_DIR, "07_confusion_matrix_dt.png")
    plt.savefig(p_cm, dpi=300)
    plt.close()
    
    results = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }
    return results, (X_train, X_test, y_train, y_test, feature_names)


def run_task_d(df):
    print("\n" + "=" * 70)
    print("TASK D: K-MEANS — CUSTOMER SEGMENTATION")
    print("=" * 70)
    
    # Select features
    cluster_features = ['Annual_Income', 'Spending_Score']
    X_cluster = df[cluster_features].copy()
    
    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "kmeans_scaler.pkl"))
    
    # Elbow method: Inertia for K=1 to 10
    inertia_list = []
    silhouette_list = []
    K_range = range(1, 11)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia_list.append(km.inertia_)
        if k >= 2:
            sil = silhouette_score(X_scaled, km.labels_)
            silhouette_list.append(sil)
            
    # Plot Elbow Curve and Silhouette Scores
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(K_range, inertia_list, marker='o', color='#2980b9', linewidth=2, label='Inertia (Elbow Curve)')
    ax1.set_xlabel("Number of Clusters (K)", fontweight='bold')
    ax1.set_ylabel("Within-Cluster Sum of Squares (Inertia)", color='#2980b9', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    ax1.set_xticks(list(K_range))
    
    ax2 = ax1.twinx()
    ax2.plot(range(2, 11), silhouette_list, marker='s', color='#27ae60', linestyle='--', linewidth=2, label='Silhouette Score')
    ax2.set_ylabel("Silhouette Score", color='#27ae60', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    
    plt.title("K-Means Optimal K Selection: Elbow Method & Silhouette Scores", fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    p_elbow = os.path.join(PLOTS_DIR, "08_kmeans_elbow_plot.png")
    plt.savefig(p_elbow, dpi=300)
    plt.close()
    
    # Fit selected K = 4
    optimal_k = 4
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    df['Cluster'] = kmeans_optimal.fit_predict(X_scaled)
    joblib.dump(kmeans_optimal, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    
    # Centroids in original units
    centroids_scaled = kmeans_optimal.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    
    print(f"\nOptimal K selected: {optimal_k}")
    print("\nCluster Centroids (Original Units):")
    for i, c in enumerate(centroids_original):
        print(f"Cluster {i}: Annual Income = INR {c[0]:.1f}k, Spending Score = {c[1]:.1f}")
        
    # Profile clusters with average income, spending score, and purchase frequency
    cluster_profiles = df.groupby('Cluster').agg(
        Count=('Customer_ID', 'count'),
        Avg_Income=('Annual_Income', 'mean'),
        Avg_Spending_Score=('Spending_Score', 'mean'),
        Avg_Purchase_Freq=('Purchase_Frequency', 'mean'),
        Campaign_Response_Rate=('Campaign_Response', lambda x: (x == 'Yes').mean() * 100)
    ).reset_index()
    
    # Assign business personas based on quadrant profiles
    persona_mapping = {}
    for idx, row in cluster_profiles.iterrows():
        c_id = row['Cluster']
        inc = row['Avg_Income']
        score = row['Avg_Spending_Score']
        if inc >= 600 and score >= 50:
            label = "Affluent Spenders (VIPs)"
        elif inc >= 600 and score < 50:
            label = "Careful Savers (High Income, Low Spend)"
        elif inc < 600 and score >= 50:
            label = "Aspirational Shoppers (Low Income, High Spend)"
        else:
            label = "Budget Seekers (Low Income, Low Spend)"
        persona_mapping[c_id] = label
        
    df['Cluster_Persona'] = df['Cluster'].map(persona_mapping)
    cluster_profiles['Persona'] = cluster_profiles['Cluster'].map(persona_mapping)
    
    print("\nCustomer Cluster Profiles:")
    print(cluster_profiles[['Cluster', 'Persona', 'Count', 'Avg_Income', 'Avg_Spending_Score', 'Avg_Purchase_Freq', 'Campaign_Response_Rate']])
    
    # Plot K-Means Clusters in Original Units
    plt.figure(figsize=(9, 6))
    for c_id in range(optimal_k):
        subset = df[df['Cluster'] == c_id]
        persona = persona_mapping[c_id]
        plt.scatter(
            subset['Annual_Income'],
            subset['Spending_Score'],
            s=60,
            alpha=0.75,
            label=f"Cluster {c_id}: {persona} (n={len(subset)})"
        )
        
    # Plot centroids
    plt.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        s=250,
        c='red',
        marker='X',
        edgecolor='black',
        linewidth=1.5,
        label='Centroids'
    )
    
    plt.title("K-Means Customer Segmentation (Income vs Spending Score)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Annual Income (INR '000)", fontweight='bold')
    plt.ylabel("Spending Score (1–100)", fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    p_clusters = os.path.join(PLOTS_DIR, "09_kmeans_clusters.png")
    plt.savefig(p_clusters, dpi=300)
    plt.close()
    
    results = {
        'optimal_k': optimal_k,
        'centroids': centroids_original.tolist(),
        'profiles': cluster_profiles.to_dict(orient='records')
    }
    return results, df


def run_task_e(split_data):
    print("\n" + "=" * 70)
    print("TASK E: RANDOM FOREST — CAMPAIGN RESPONSE PREDICTION")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test, feature_names = split_data
    
    # Train Random Forest on identical train/test split
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nRandom Forest Test Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
    
    # Save model artifact
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.pkl"))
    
    # Feature Importances
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    print("\nFeature Importances:")
    print(fi_df.sort_values('Importance', ascending=False))
    
    # Plot Feature Importance
    plt.figure(figsize=(8, 5))
    plt.barh(fi_df['Feature'], fi_df['Importance'], color='#16a085', edgecolor='black')
    plt.title("Random Forest: Feature Importance (Campaign Response)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Gini Importance (Relative Predictive Weight)")
    plt.tight_layout()
    p_fi = os.path.join(PLOTS_DIR, "10_feature_importance_rf.png")
    plt.savefig(p_fi, dpi=300)
    plt.close()
    
    results = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'feature_importances': fi_df.to_dict(orient='records')
    }
    return results, rf_model


def run_model_comparison(dt_results, rf_results):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: DECISION TREE VS RANDOM FOREST")
    print("=" * 70)
    
    comp_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Decision Tree': [
            dt_results['accuracy'],
            dt_results['precision'],
            dt_results['recall'],
            dt_results['f1']
        ],
        'Random Forest': [
            rf_results['accuracy'],
            rf_results['precision'],
            rf_results['recall'],
            rf_results['f1']
        ]
    })
    print(comp_df.to_string(index=False))
    
    # Plot side-by-side comparison
    df_melt = comp_df.melt(id_vars='Metric', var_name='Model', value_name='Score')
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette=['#3498db', '#2ecc71'])
    plt.title("Model Comparison: Decision Tree vs. Random Forest", fontsize=13, fontweight='bold', pad=12)
    plt.ylim(0, 1.1)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    p_comp = os.path.join(PLOTS_DIR, "11_model_comparison_dt_vs_rf.png")
    plt.savefig(p_comp, dpi=300)
    plt.close()
    
    return comp_df


def main():
    print("Starting BrightMart Analytical Machine Learning Pipeline...")
    
    # Task A
    df_clean = run_task_a()
    
    # Task B
    reg_results, reg_splits = run_task_b(df_clean)
    
    # Task C
    dt_results, clf_splits = run_task_c(df_clean)
    
    # Task D
    kmeans_results, df_clustered = run_task_d(df_clean)
    
    # Task E
    rf_results, rf_model = run_task_e(clf_splits)
    
    # Model Comparison
    comp_df = run_model_comparison(dt_results, rf_results)
    
    # Save all metrics to JSON
    all_metrics = {
        'task_a': {
            'cleaned_rows': len(df_clean),
            'cleaned_cols': len(df_clean.columns)
        },
        'task_b_regression': reg_results,
        'task_c_decision_tree': dt_results,
        'task_d_kmeans': kmeans_results,
        'task_e_random_forest': rf_results
    }
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"\nAll task metrics saved to: {METRICS_PATH}")
    print("Pipeline completed successfully! All plots and models are ready.")


if __name__ == "__main__":
    main()
