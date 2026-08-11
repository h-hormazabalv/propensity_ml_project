# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Baseline ML-Propensity Models - Classification Approach
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Notebook Overview
# MAGIC %md
# MAGIC ## Binary Classification Framework - Multi-Model Comparison
# MAGIC
# MAGIC ## Purpose
# MAGIC This notebook serves as a **reusable template** for binary classification problems. It's designed to accept any post-feature-engineering dataset and automatically benchmark four ML algorithms with minimal configuration, just specify your target variable and optional parameters.
# MAGIC
# MAGIC **Current Example**: Churn propensity modeling (58.1% positive class)
# MAGIC
# MAGIC ## Key Features
# MAGIC * **Plug-and-Play Architecture**: Works with any binary classification dataset after feature engineering
# MAGIC * **Automated Feature Selection**: SelectFromModel with configurable threshold
# MAGIC * **Hyperparameter Optimization**: Optuna with TPE sampler, early pruning, and parallel trials
# MAGIC * **Rigorous Validation**: 5-fold Stratified CV optimizing F1-score
# MAGIC * **Threshold Calibration**: Optimal decision boundary tuned on precision-recall curve
# MAGIC * **Production-Ready Metrics**: Precision, Recall, F1, AUC-ROC, AU-PR, Accuracy
# MAGIC * **Model Explainability**: Native feature importance + SHAP value analysis
# MAGIC
# MAGIC ## Models Benchmarked
# MAGIC 1. **XGBoost** - Gradient boosting with depth/rate tuning
# MAGIC 2. **LightGBM** - Leaf-wise gradient boosting
# MAGIC 3. **Decision Tree** - Interpretable classifier with pruning
# MAGIC 4. **Logistic Regression** - ElasticNet (L1+L2 regularization)
# MAGIC
# MAGIC ## Output
# MAGIC Automated ranking by **Test F1-Score** with comprehensive performance dashboard, optimal hyperparameters, and actionable recommendations.
# MAGIC
# MAGIC ---
# MAGIC **Roadmap**: Additional algorithms will be added in future iterations to expand the benchmarking suite.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main imports
# MAGIC ---

# COMMAND ----------

# MAGIC %pip install -q xgboost lightgbm shap optuna

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree, export_text
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix
)
import shap
import optuna

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data reading
# MAGIC ---

# COMMAND ----------

path = "workspace.personal_playground.ecommerce_data_cleaned"
df   = spark.table(path).toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

# DBTITLE 1,Global Train/Test Split
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = 'churn'
ID_COL = 'customer_id'

cols_to_drop = [TARGET_COL, ID_COL]
X = df.drop(columns=cols_to_drop)
y = df[TARGET_COL]

# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("=" * 70)
print(" GLOBAL TRAIN/TEST SPLIT (Shared Across All Models)")
print("=" * 70)
print(f"Random State:      {RANDOM_STATE}")
print(f"Test Size:         {TEST_SIZE:.0%}")
print(f"\nTrain Set:         {X_train.shape[0]:,} records | {X_train.shape[1]} features")
print(f"Test Set:          {X_test.shape[0]:,} records | {X_test.shape[1]} features")

num_neg_train = (y_train == 0).sum()
num_pos_train = (y_train == 1).sum()
num_neg_test = (y_test == 0).sum()
num_pos_test = (y_test == 1).sum()

print(f"\nClass Distribution (Train):")
print(f"  - Negative (0):  {num_neg_train:,} ({num_neg_train/len(y_train):.1%})")
print(f"  - Positive (1):  {num_pos_train:,} ({num_pos_train/len(y_train):.1%})")

print(f"\nClass Distribution (Test):")
print(f"  - Negative (0):  {num_neg_test:,} ({num_neg_test/len(y_test):.1%})")
print(f"  - Positive (1):  {num_pos_test:,} ({num_pos_test/len(y_test):.1%})")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGBoost
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,XGBoost pipeline
def fit_xgb_churn_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    df: pd.DataFrame,
    target_col: str = 'churn',
    id_col: str = 'customer_id',
    threshold_fs: str = 'median',
    n_iter_search: int = 15,
    cv_splits: int = 5,
    random_state: int = 42
) -> dict:
    

    print(" 1. DATASET INFORMATION AND IMBALANCE RATIO")
    print("=" * 60)
    print(f"- Train set: {X_train.shape[0]} records | {X_train.shape[1]} features")
    print(f"- Test set:  {X_test.shape[0]} records | {X_test.shape[1]} features")

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"- Calculated scale_pos_weight ratio: {scale_pos_weight:.3f}")

    # MODEL PIPELINE 
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='aucpr',
        n_jobs=-1
    )

    selector = SelectFromModel(
        estimator=xgb_base,
        threshold=threshold_fs,
        importance_getter='feature_importances_'
    )

    pipeline = Pipeline([
        ('feature_selection', selector),
        ('xgb', xgb_base)
    ])

    # HYPERPARAMETER TUNING WITH OPTUNA
    print("\n")
    print(" 2. OPTIMIZATION AND CROSS-VALIDATION (OPTUNA)")
    print("=" * 60)
    
    # CROSS VALIDATION STRATEGY
    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # OPTUNA OBJECTIVE FUNCTION
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
            'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 9]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.8, 1.0]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.8, 1.0]),
            'min_child_weight': trial.suggest_categorical('min_child_weight', [1, 3, 5]),
            'gamma': trial.suggest_categorical('gamma', [0, 0.1, 0.2])
        }
        
        # CREATE MODEL
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='aucpr',
            n_jobs=-1,
            **params
        )
        
        # CREATE PIPELINE WITH FEATURE SELECTION
        selector = SelectFromModel(
            estimator=model,
            threshold=threshold_fs,
            importance_getter='feature_importances_'
        )
        
        trial_pipeline = Pipeline([
            ('feature_selection', selector),
            ('xgb', model)
        ])
        
        # MANUAL CV LOOP FOR PRUNING
        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            trial_pipeline.fit(X_fold_train, y_fold_train)
            fold_score = f1_score(y_fold_val, trial_pipeline.predict(X_fold_val), zero_division=0)
            cv_scores.append(fold_score)
            
            trial.report(np.mean(cv_scores), fold_idx)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(cv_scores)
    
    # CREATE OPTUNA STUDY AND OPTIMIZE
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='xgboost_propensity_optimization'
    )
    
    study.optimize(objective, n_trials=n_iter_search, show_progress_bar=True)
    
    # EXTRACT BEST HYPERPARAMETERS
    best_params = study.best_params
    cv_f1_mean = study.best_value
    best_trial = study.best_trial
    cv_f1_std = 0.0
    
    # PRUNING STATISTICS
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print("\n--- Cross-Validation Results (F1-Score Metric) ---")
    print(f" - Best CV F1-Score: {cv_f1_mean:.4f}")
    print(f" - Best Trial Number: {best_trial.number}")
    print(f" - Total Trials Started: {len(study.trials)}")
    print(f" - Trials Completed: {len(completed_trials)}")
    print(f" - Trials Pruned (Early Stopping): {len(pruned_trials)}")
    if len(pruned_trials) > 0:
        print(f" - Time Saved by Pruning: ~{len(pruned_trials) * 10}s (estimated)")
    
    print("\n--- Optimal Hyperparameters Found ---")
    cleaned_params = best_params.copy()
    for param, value in cleaned_params.items():
        print(f" - {param}: {value}")
    
    # TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
    best_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='aucpr',
        n_jobs=-1,
        **best_params
    )
    
    best_selector = SelectFromModel(
        estimator=best_model,
        threshold=threshold_fs,
        importance_getter='feature_importances_'
    )
    
    best_pipeline = Pipeline([
        ('feature_selection', best_selector),
        ('xgb', best_model)
    ])
    
    best_pipeline.fit(X_train, y_train)
    best_model = best_pipeline.named_steps['xgb']
    best_selector = best_pipeline.named_steps['feature_selection']

    # FEATURE SELECTION REPORTING
    selected_mask = best_selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    removed_features = X.columns[~selected_mask].tolist()

    print("\n")
    print(" 3. FEATURE SELECTION")
    print("=" * 60)
    print(f" - Initial Predictor Features: {X.shape[1]}")
    print(f" - Selected Features:          {len(selected_features)}")
    print(f" - Discarded Features:         {len(removed_features)}")

    print("\n--- Kept Features List ---")
    for feat in selected_features:
        print(f"   - {feat}")

    print("\n--- Discarded Features List ---")
    for feat in removed_features:
        print(f"   - {feat}")

    # TRANSFORMATION
    X_train_sel = pd.DataFrame(best_selector.transform(X_train), columns=selected_features, index=X_train.index)
    X_test_sel = pd.DataFrame(best_selector.transform(X_test), columns=selected_features, index=X_test.index)

    # PROBABILITIES AND OPTIMAL THRESHOLD
    probs_train = best_model.predict_proba(X_train_sel)[:, 1]
    probs_test = best_model.predict_proba(X_test_sel)[:, 1]

    precisions_tr, recalls_tr, thresholds_tr = precision_recall_curve(y_train, probs_train)
    f1_scores_tr = (2 * precisions_tr * recalls_tr) / (precisions_tr + recalls_tr + 1e-8)
    best_idx = np.argmax(f1_scores_tr)
    best_threshold = thresholds_tr[best_idx] if best_idx < len(thresholds_tr) else 0.5

    print("\n")
    print(" 4. EVALUATION AND OPTIMAL THRESHOLD")
    print("=" * 60)
    print(f"- Optimal Threshold for F1-Score: {best_threshold:.4f}")

    preds_train_opt = (probs_train >= best_threshold).astype(int)
    preds_test_opt = (probs_test >= best_threshold).astype(int)

    # OUTPUT DATAFRAMES
    pred_prob_col = f'{target_col}_pred_prob'
    pred_label_col = f'{target_col}_pred_label_optimal'
    
    df_train_out = df.loc[X_train.index].copy()
    df_train_out[pred_prob_col] = probs_train
    df_train_out[pred_label_col] = preds_train_opt

    df_test_out = df.loc[X_test.index].copy()
    df_test_out[pred_prob_col] = probs_test
    df_test_out[pred_label_col] = preds_test_opt

    front_cols = [col for col in [id_col, target_col, pred_prob_col, pred_label_col] if col in df_train_out.columns]
    other_cols = [col for col in df_train_out.columns if col not in front_cols]
    
    df_train_out = df_train_out[front_cols + other_cols]
    df_test_out = df_test_out[front_cols + other_cols]

    # TRAIN/TEST METRICS
    def calculate_metrics(y_true, probs, preds, dataset_name):
        return {
            'Dataset': dataset_name,
            'Accuracy': accuracy_score(y_true, preds),
            'Precision': precision_score(y_true, preds, zero_division=0),
            'Recall': recall_score(y_true, preds, zero_division=0),
            'F1-Score (Opt)': f1_score(y_true, preds, zero_division=0),
            'AUC-ROC': roc_auc_score(y_true, probs),
            'AU-PR': average_precision_score(y_true, probs)
        }

    metrics_df = pd.DataFrame([
        calculate_metrics(y_train, probs_train, preds_train_opt, 'Train'),
        calculate_metrics(y_test, probs_test, preds_test_opt, 'Test')
    ]).set_index('Dataset')

    print("\n--- Final Performance Summary ---")
    print(metrics_df.round(4))

    # VISUALIZATIONS: ROC, PR CURVES AND CONFUSION MATRIX
    fig_eval, axes_eval = plt.subplots(1, 3, figsize=(18, 5))

    # ROC CURVE
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    auc_roc = roc_auc_score(y_test, probs_test)
    
    axes_eval[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_roc:.4f})')
    axes_eval[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    axes_eval[0].set_xlim([0.0, 1.0])
    axes_eval[0].set_ylim([0.0, 1.05])
    axes_eval[0].set_xlabel('False Positive Rate')
    axes_eval[0].set_ylabel('True Positive Rate')
    axes_eval[0].set_title('ROC Curve - XGBoost (Test Set)', fontsize=12, fontweight='bold')
    axes_eval[0].legend(loc="lower right")
    axes_eval[0].grid(True, linestyle='--', alpha=0.6)

    # PRECISION-RECALL CURVE
    prec_test, rec_test, _ = precision_recall_curve(y_test, probs_test)
    au_pr = average_precision_score(y_test, probs_test)
    
    axes_eval[1].plot(rec_test, prec_test, color='blue', lw=2, label=f'PR (AU-PR = {au_pr:.4f})')
    prec_opt = precision_score(y_test, preds_test_opt, zero_division=0)
    rec_opt = recall_score(y_test, preds_test_opt, zero_division=0)
    axes_eval[1].plot(rec_opt, prec_opt, 'ro', markersize=8, label=f'Threshold ({best_threshold:.3f})')
    
    axes_eval[1].set_xlim([0.0, 1.0])
    axes_eval[1].set_ylim([0.0, 1.05])
    axes_eval[1].set_xlabel('Recall (Sensitivity)')
    axes_eval[1].set_ylabel('Precision')
    axes_eval[1].set_title('Precision-Recall Curve - XGBoost (Test Set)', fontsize=12, fontweight='bold')
    axes_eval[1].legend(loc="lower left")
    axes_eval[1].grid(True, linestyle='--', alpha=0.6)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, preds_test_opt)
    cm_norm = confusion_matrix(y_test, preds_test_opt, normalize='true')
    
    labels = np.array([[f"{val:,}\n({norm:.1%})" for val, norm in zip(row_cm, row_norm)]
                       for row_cm, row_norm in zip(cm, cm_norm)])

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=axes_eval[2],
                xticklabels=['No Propensity (0)', 'Propensity (1)'],
                yticklabels=['No Propensity (0)', 'Propensity (1)'])
    axes_eval[2].set_xlabel('Model Prediction')
    axes_eval[2].set_ylabel('True Value')
    axes_eval[2].set_title(f'Confusion Matrix - XGBoost (Test Set)\nThreshold: {best_threshold:.4f}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('evaluation_metrics_summary.png', dpi=150)
    plt.show()

    # FEATURE IMPORTANCE VISUALIZATIONS (WEIGHT AND GAIN)
    booster = best_model.get_booster()
    booster.feature_names = selected_features

    weight_imp = pd.Series(booster.get_score(importance_type='weight'), name='Weight')
    gain_imp = pd.Series(booster.get_score(importance_type='gain'), name='Gain')

    feat_imp = pd.DataFrame({'Weight': weight_imp, 'Gain': gain_imp}).fillna(0)
    feat_imp.index.name = 'Feature'

    fig_imp, axes_imp = plt.subplots(1, 2, figsize=(14, 5))
    feat_imp.sort_values(by='Weight', ascending=True)['Weight'].tail(15).plot(
        kind='barh', ax=axes_imp[0], color='skyblue'
    )
    axes_imp[0].set_title('XGBoost Feature Relevance (Weight / Freq)')
    axes_imp[0].set_xlabel('Split Frequency')

    feat_imp.sort_values(by='Gain', ascending=True)['Gain'].tail(15).plot(
        kind='barh', ax=axes_imp[1], color='salmon'
    )
    axes_imp[1].set_title('XGBoost Information Gain')
    axes_imp[1].set_xlabel('Mean Gain')
    plt.tight_layout()
    plt.savefig('xgb_native_feature_importance.png', dpi=150)
    plt.show()

    # 9. SHAP VISUALIZATIONS
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_test_sel)

    plt.figure(figsize=(10, 6))
    plt.title('XGBoost SHAP Summary Plot (Selected Features)', fontsize=12)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=150)
    plt.show()
    
    print("\n")
    plt.figure(figsize=(10, 6))
    plt.title('XGBoost SHAP Feature Importance (Bar Plot)', fontsize=12)
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('shap_bar_importance.png', dpi=150)
    plt.show()

    return {
        'best_pipeline': best_pipeline,
        'best_model': best_model,
        'best_params': cleaned_params,
        'cv_f1_score_mean': cv_f1_mean,
        'cv_f1_score_std': cv_f1_std,
        'optuna_study': study,
        'selected_features': selected_features,
        'removed_features': removed_features,
        'optimal_threshold': best_threshold,
        'metrics': metrics_df,
        'df_train_results': df_train_out,
        'df_test_results': df_test_out,
        'feature_importance': feat_imp,
        'shap_values': shap_values,
        'explainer': explainer
    }

# COMMAND ----------

# DBTITLE 1,XGBoost results
results_xgboost = fit_xgb_churn_pipeline(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    df=df,
    target_col='churn',
    id_col='customer_id',
    n_iter_search=20
)

df_test_predictions = results_xgboost['df_test_results']

target_col = 'churn'
pred_prob_col = f'{target_col}_pred_prob'
pred_label_col = f'{target_col}_pred_label_optimal'

display(df_test_predictions[['customer_id', target_col, pred_prob_col, pred_label_col]].head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## LightGBM
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,LGBM pipeline
def fit_lgb_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    df: pd.DataFrame,
    target_col: str = 'churn',
    id_col: str = 'customer_id',
    threshold_fs: str = 'median',
    n_iter_search: int = 15,
    cv_splits: int = 5,
    random_state: int = 42
) -> dict:
    
  
    print(" 1. DATASET INFORMATION AND IMBALANCE RATIO")
    print("=" * 60)
    print(f"- Train set: {X_train.shape[0]} records | {X_train.shape[1]} features")
    print(f"- Test set:  {X_test.shape[0]} records | {X_test.shape[1]} features")

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"- Calculated scale_pos_weight ratio: {scale_pos_weight:.3f}")

    # MODEL PIPELINE 
    lgb_base = lgb.LGBMClassifier(
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )

    selector = SelectFromModel(
        estimator=lgb_base,
        threshold=threshold_fs,
        importance_getter='feature_importances_'
    )

    pipeline = Pipeline([
        ('feature_selection', selector),
        ('lgb', lgb_base)
    ])

    # HYPERPARAMETER TUNING WITH OPTUNA
    print("\n")
    print(" 2. OPTIMIZATION AND CROSS-VALIDATION (OPTUNA)")
    print("=" * 60)
    
    # CROSS VALIDATION STRATEGY
    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # OPTUNA OBJECTIVE FUNCTION
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
            'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63, 127]),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 3, 5, 7, 9]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.8, 1.0]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.8, 1.0]),
            'min_child_samples': trial.suggest_categorical('min_child_samples', [10, 20, 30, 50]),
            'reg_alpha': trial.suggest_categorical('reg_alpha', [0.0, 0.1, 1.0]),
            'reg_lambda': trial.suggest_categorical('reg_lambda', [0.0, 0.1, 1.0])
        }
        
        # CREATE MODEL
        model = lgb.LGBMClassifier(
            objective='binary',
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **params
        )
        
        # CREATE PIPELINE WITH FEATURE SELECTION
        selector = SelectFromModel(
            estimator=model,
            threshold=threshold_fs,
            importance_getter='feature_importances_'
        )
        
        trial_pipeline = Pipeline([
            ('feature_selection', selector),
            ('lgb', model)
        ])
        
        # MANUAL CV LOOP FOR PRUNING
        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            trial_pipeline.fit(X_fold_train, y_fold_train)
            fold_score = f1_score(y_fold_val, trial_pipeline.predict(X_fold_val), zero_division=0)
            cv_scores.append(fold_score)
            
            trial.report(np.mean(cv_scores), fold_idx)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(cv_scores)
    
    # CREATE OPTUNA STUDY AND OPTIMIZE
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='lightgbm_propensity_optimization'
    )
    
    study.optimize(objective, n_trials=n_iter_search, show_progress_bar=True)
    
    # EXTRACT BEST HYPERPARAMETERS
    best_params = study.best_params
    cv_f1_mean = study.best_value
    best_trial = study.best_trial
    cv_f1_std = 0.0
    
    # PRUNING STATISTICS
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print("\n--- Cross-Validation Results (F1-Score Metric) ---")
    print(f" - Best CV F1-Score: {cv_f1_mean:.4f}")
    print(f" - Best Trial Number: {best_trial.number}")
    print(f" - Total Trials Started: {len(study.trials)}")
    print(f" - Trials Completed: {len(completed_trials)}")
    print(f" - Trials Pruned (Early Stopping): {len(pruned_trials)}")
    if len(pruned_trials) > 0:
        print(f" - Time Saved by Pruning: ~{len(pruned_trials) * 60}s (estimated)")
    
    print("\n--- Optimal Hyperparameters Found ---")
    cleaned_params = best_params.copy()
    for param, value in cleaned_params.items():
        print(f" - {param}: {value}")
    
    # TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
    best_model = lgb.LGBMClassifier(
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        **best_params
    )
    
    best_selector = SelectFromModel(
        estimator=best_model,
        threshold=threshold_fs,
        importance_getter='feature_importances_'
    )
    
    best_pipeline = Pipeline([
        ('feature_selection', best_selector),
        ('lgb', best_model)
    ])
    
    best_pipeline.fit(X_train, y_train)
    best_model = best_pipeline.named_steps['lgb']
    best_selector = best_pipeline.named_steps['feature_selection']

    # FEATURE SELECTION REPORTING
    selected_mask = best_selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    removed_features = X.columns[~selected_mask].tolist()

    print("\n")
    print(" 3. FEATURE SELECTION")
    print("=" * 60)
    print(f" - Initial Predictor Features: {X.shape[1]}")
    print(f" - Selected Features:          {len(selected_features)}")
    print(f" - Discarded Features:         {len(removed_features)}")

    print("\n--- Kept Features List ---")
    for feat in selected_features:
        print(f"   - {feat}")

    print("\n--- Discarded Features List ---")
    for feat in removed_features:
        print(f"   - {feat}")

    # TRANSFORMATION
    X_train_sel = pd.DataFrame(best_selector.transform(X_train), columns=selected_features, index=X_train.index)
    X_test_sel = pd.DataFrame(best_selector.transform(X_test), columns=selected_features, index=X_test.index)

    # PROBABILITIES AND OPTIMAL THRESHOLD
    probs_train = best_model.predict_proba(X_train_sel)[:, 1]
    probs_test = best_model.predict_proba(X_test_sel)[:, 1]

    precisions_tr, recalls_tr, thresholds_tr = precision_recall_curve(y_train, probs_train)
    f1_scores_tr = (2 * precisions_tr * recalls_tr) / (precisions_tr + recalls_tr + 1e-8)
    best_idx = np.argmax(f1_scores_tr)
    best_threshold = thresholds_tr[best_idx] if best_idx < len(thresholds_tr) else 0.5

    print("\n" + "=" * 60)
    print(" 4. EVALUATION AND OPTIMAL THRESHOLD")
    print("=" * 60)
    print(f"- Optimal Threshold for F1-Score: {best_threshold:.4f}")

    preds_train_opt = (probs_train >= best_threshold).astype(int)
    preds_test_opt = (probs_test >= best_threshold).astype(int)

    # OUTPUT DATAFRAMES
    pred_prob_col = f'{target_col}_pred_prob'
    pred_label_col = f'{target_col}_pred_label_optimal'
    
    df_train_out = df.loc[X_train.index].copy()
    df_train_out[pred_prob_col] = probs_train
    df_train_out[pred_label_col] = preds_train_opt

    df_test_out = df.loc[X_test.index].copy()
    df_test_out[pred_prob_col] = probs_test
    df_test_out[pred_label_col] = preds_test_opt

    front_cols = [col for col in [id_col, target_col, pred_prob_col, pred_label_col] if col in df_train_out.columns]
    other_cols = [col for col in df_train_out.columns if col not in front_cols]
    
    df_train_out = df_train_out[front_cols + other_cols]
    df_test_out = df_test_out[front_cols + other_cols]

    # TRAIN/TEST METRICS
    def calculate_metrics(y_true, probs, preds, dataset_name):
        return {
            'Dataset': dataset_name,
            'Accuracy': accuracy_score(y_true, preds),
            'Precision': precision_score(y_true, preds, zero_division=0),
            'Recall': recall_score(y_true, preds, zero_division=0),
            'F1-Score (Opt)': f1_score(y_true, preds, zero_division=0),
            'AUC-ROC': roc_auc_score(y_true, probs),
            'AU-PR': average_precision_score(y_true, probs)
        }

    metrics_df = pd.DataFrame([
        calculate_metrics(y_train, probs_train, preds_train_opt, 'Train'),
        calculate_metrics(y_test, probs_test, preds_test_opt, 'Test')
    ]).set_index('Dataset')

    print("\n--- Final Performance Summary ---")
    print(metrics_df.round(4))

    # VISUALIZATIONS: ROC, PR CURVES AND CONFUSION MATRIX
    fig_eval, axes_eval = plt.subplots(1, 3, figsize=(18, 5))

    # ROC CURVE
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    auc_roc = roc_auc_score(y_test, probs_test)
    
    axes_eval[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_roc:.4f})')
    axes_eval[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    axes_eval[0].set_xlim([0.0, 1.0])
    axes_eval[0].set_ylim([0.0, 1.05])
    axes_eval[0].set_xlabel('False Positive Rate')
    axes_eval[0].set_ylabel('True Positive Rate')
    axes_eval[0].set_title('ROC Curve - LightGBM (Test Set)', fontsize=12, fontweight='bold')
    axes_eval[0].legend(loc="lower right")
    axes_eval[0].grid(True, linestyle='--', alpha=0.6)

    # PRECISION-RECALL CURVE
    prec_test, rec_test, _ = precision_recall_curve(y_test, probs_test)
    au_pr = average_precision_score(y_test, probs_test)
    
    axes_eval[1].plot(rec_test, prec_test, color='blue', lw=2, label=f'PR (AU-PR = {au_pr:.4f})')
    prec_opt = precision_score(y_test, preds_test_opt, zero_division=0)
    rec_opt = recall_score(y_test, preds_test_opt, zero_division=0)
    axes_eval[1].plot(rec_opt, prec_opt, 'ro', markersize=8, label=f'Threshold ({best_threshold:.3f})')
    
    axes_eval[1].set_xlim([0.0, 1.0])
    axes_eval[1].set_ylim([0.0, 1.05])
    axes_eval[1].set_xlabel('Recall (Sensitivity)')
    axes_eval[1].set_ylabel('Precision')
    axes_eval[1].set_title('Precision-Recall Curve - LightGBM (Test Set)', fontsize=12, fontweight='bold')
    axes_eval[1].legend(loc="lower left")
    axes_eval[1].grid(True, linestyle='--', alpha=0.6)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, preds_test_opt)
    cm_norm = confusion_matrix(y_test, preds_test_opt, normalize='true')
    
    labels = np.array([[f"{val:,}\n({norm:.1%})" for val, norm in zip(row_cm, row_norm)]
                       for row_cm, row_norm in zip(cm, cm_norm)])

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=axes_eval[2],
                xticklabels=['No Propensity (0)', 'Propensity (1)'],
                yticklabels=['No Propensity (0)', 'Propensity (1)'])
    axes_eval[2].set_xlabel('Model Prediction')
    axes_eval[2].set_ylabel('True Value')
    axes_eval[2].set_title(f'Confusion Matrix - LightGBM (Test Set)\nThreshold: {best_threshold:.4f}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('evaluation_metrics_summary.png', dpi=150)
    plt.show()

    # FEATURE IMPORTANCE VISUALIZATIONS (SPLIT / WEIGHT AND GAIN)
    booster = best_model.booster_
    
    split_imp = pd.Series(booster.feature_importance(importance_type='split'), index=selected_features, name='Weight')
    gain_imp = pd.Series(booster.feature_importance(importance_type='gain'), index=selected_features, name='Gain')

    feat_imp = pd.DataFrame({'Weight': split_imp, 'Gain': gain_imp}).fillna(0)
    feat_imp.index.name = 'Feature'

    fig_imp, axes_imp = plt.subplots(1, 2, figsize=(14, 5))
    feat_imp.sort_values(by='Weight', ascending=True)['Weight'].tail(15).plot(
        kind='barh', ax=axes_imp[0], color='skyblue'
    )
    axes_imp[0].set_title('LightGBM Feature Relevance (Split Frequency)')
    axes_imp[0].set_xlabel('Split Frequency')

    feat_imp.sort_values(by='Gain', ascending=True)['Gain'].tail(15).plot(
        kind='barh', ax=axes_imp[1], color='salmon'
    )
    axes_imp[1].set_title('LightGBM Information Gain')
    axes_imp[1].set_xlabel('Mean Gain')
    plt.tight_layout()
    plt.savefig('lgb_native_feature_importance.png', dpi=150)
    plt.show()

    # SHAP VISUALIZATIONS
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_test_sel)

    plt.figure(figsize=(10, 6))
    plt.title('LightGBM SHAP Summary Plot (Selected Features)', fontsize=12)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=150)
    plt.show()

    print("\n")


    plt.figure(figsize=(10, 6))
    plt.title('LightGBM SHAP Feature Importance (Bar Plot)', fontsize=12)
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('shap_bar_importance.png', dpi=150)
    plt.show()

    return {
        'best_pipeline': best_pipeline,
        'best_model': best_model,
        'best_params': cleaned_params,
        'cv_f1_score_mean': cv_f1_mean,
        'cv_f1_score_std': cv_f1_std,
        'optuna_study': study,
        'selected_features': selected_features,
        'removed_features': removed_features,
        'optimal_threshold': best_threshold,
        'metrics': metrics_df,
        'df_train_results': df_train_out,
        'df_test_results': df_test_out,
        'feature_importance': feat_imp,
        'shap_values': shap_values,
        'explainer': explainer
    }

# COMMAND ----------

# DBTITLE 1,LGBM results
results_lgbm = fit_lgb_pipeline(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    df=df,
    target_col='churn',
    id_col='customer_id',
    n_iter_search=20
)
 

df_test_predictions = results_lgbm['df_test_results']

target_col = 'churn'
pred_prob_col = f'{target_col}_pred_prob'
pred_label_col = f'{target_col}_pred_label_optimal'

display(df_test_predictions[['customer_id', target_col, pred_prob_col, pred_label_col]].head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Decision Tree pipeline
def fit_dt_churn_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    df: pd.DataFrame,
    target_col: str = 'churn',
    id_col: str = 'customer_id',
    class_weight: str | dict | None = 'balanced',
    threshold_fs: str = 'median',
    n_iter_search: int = 15,
    cv_splits: int = 5,
    max_depth_plot: int | None = 3,
    random_state: int = 42
) -> dict:

    print(" 1. DATASET INFORMATION AND IMBALANCE RATIO")
    print("=" * 60)
    print(f"- Model Algorithm: Decision Tree Classifier")
    print(f"- Train set: {X_train.shape[0]} records | {X_train.shape[1]} features")
    print(f"- Test set:  {X_test.shape[0]} records | {X_test.shape[1]} features")

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    imbalance_ratio = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"- Class distribution in Train -> Negative (0): {num_neg:,} | Positive (1): {num_pos:,}")
    print(f"- Calculated imbalance ratio (neg/pos): {imbalance_ratio:.2f}:1")
    print(f"- Decision Tree Class weight strategy: Optimized by Optuna")

    # BASE DECISION TREE MODEL & SELECTOR
    dt_base = DecisionTreeClassifier(
        class_weight=class_weight,
        random_state=random_state
    )

    selector = SelectFromModel(
        estimator=dt_base,
        threshold=threshold_fs,
        importance_getter='feature_importances_'
    )

    pipeline = Pipeline([
        ('feature_selection', selector),
        ('dt', dt_base)
    ])

    # HYPERPARAMETER TUNING WITH OPTUNA
    print("\n")
    print(" 2. OPTIMIZATION AND CROSS-VALIDATION (OPTUNA)")
    print("=" * 60)
    
    # CROSS VALIDATION STRATEGY
    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # OPTUNA OBJECTIVE FUNCTION
    def objective(trial):
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 8, 10]),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [10, 20, 50, 100]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [5, 10, 20, 50, 100]),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'ccp_alpha': trial.suggest_categorical('ccp_alpha', [0.0, 0.001, 0.005, 0.01, 0.02]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        
        # CREATE MODEL
        model = DecisionTreeClassifier(
            random_state=random_state,
            **params
        )
        
        # CREATE PIPELINE WITH FEATURE SELECTION
        selector = SelectFromModel(
            estimator=model,
            threshold=threshold_fs,
            importance_getter='feature_importances_'
        )
        
        trial_pipeline = Pipeline([
            ('feature_selection', selector),
            ('dt', model)
        ])
        
        # MANUAL CV LOOP FOR PRUNING
        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            trial_pipeline.fit(X_fold_train, y_fold_train)
            fold_score = f1_score(y_fold_val, trial_pipeline.predict(X_fold_val), zero_division=0)
            cv_scores.append(fold_score)
            
            trial.report(np.mean(cv_scores), fold_idx)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(cv_scores)
    
    # CREATE OPTUNA STUDY AND OPTIMIZE
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='decision_tree_propensity_optimization'
    )
    
    study.optimize(objective, n_trials=n_iter_search, show_progress_bar=True)
    
    # EXTRACT BEST HYPERPARAMETERS
    best_params = study.best_params
    cv_f1_mean = study.best_value
    best_trial = study.best_trial
    cv_f1_std = 0.0
    
    # PRUNING STATISTICS
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print("\n--- Cross-Validation Results (F1-Score Metric) ---")
    print(f" - Best CV F1-Score: {cv_f1_mean:.4f}")
    print(f" - Best Trial Number: {best_trial.number}")
    print(f" - Total Trials Started: {len(study.trials)}")
    print(f" - Trials Completed: {len(completed_trials)}")
    print(f" - Trials Pruned (Early Stopping): {len(pruned_trials)}")
    if len(pruned_trials) > 0:
        print(f" - Time Saved by Pruning: ~{len(pruned_trials) * 5}s (estimated)")
    
    print("\n--- Optimal Hyperparameters Found ---")
    cleaned_params = best_params.copy()
    for param, value in cleaned_params.items():
        print(f" - {param}: {value}")
    
    # TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS
    best_model = DecisionTreeClassifier(
        random_state=random_state,
        **best_params
    )
    
    best_selector = SelectFromModel(
        estimator=best_model,
        threshold=threshold_fs,
        importance_getter='feature_importances_'
    )
    
    best_pipeline = Pipeline([
        ('feature_selection', best_selector),
        ('dt', best_model)
    ])
    
    best_pipeline.fit(X_train, y_train)
    best_model = best_pipeline.named_steps['dt']
    best_selector = best_pipeline.named_steps['feature_selection']

    # FEATURE SELECTION REPORTING
    all_feature_names = X_train.columns.tolist()
    selected_mask = best_selector.get_support()
    selected_features = [f for f, s in zip(all_feature_names, selected_mask) if s]
    removed_features = [f for f, s in zip(all_feature_names, selected_mask) if not s]

    print("\n")
    print(" 3. FEATURE SELECTION (DECISION TREE IMPORTANCE)")
    print("=" * 60)
    print(f" - Initial Predictor Features: {len(all_feature_names)}")
    print(f" - Selected Features:          {len(selected_features)}")
    print(f" - Discarded Features:         {len(removed_features)}")

    print("\n--- Kept Features List ---")
    for feat in selected_features:
        print(f"   - {feat}")

    print("\n--- Discarded Features List ---")
    for feat in removed_features:
        print(f"   - {feat}")

    # PROBABILITIES VIA PIPELINE
    probs_train = best_pipeline.predict_proba(X_train)[:, 1]
    probs_test = best_pipeline.predict_proba(X_test)[:, 1]

    # OPTIMAL THRESHOLD
    precisions_tr, recalls_tr, thresholds_tr = precision_recall_curve(y_train, probs_train)
    f1_scores_tr = (2 * precisions_tr * recalls_tr) / (precisions_tr + recalls_tr + 1e-8)
    best_idx = np.argmax(f1_scores_tr)
    best_threshold = thresholds_tr[best_idx] if best_idx < len(thresholds_tr) else 0.5

    print("\n")
    print(" 4. DECISION TREE EVALUATION & OPTIMAL THRESHOLD")
    print("=" * 60)
    print(f"- Optimal Threshold for F1-Score (calculated on Train): {best_threshold:.4f}")

    preds_train_opt = (probs_train >= best_threshold).astype(int)
    preds_test_opt = (probs_test >= best_threshold).astype(int)

    # OUTPUT DATAFRAMES
    pred_prob_col = f'{target_col}_pred_prob'
    pred_label_col = f'{target_col}_pred_label_optimal'
    
    df_train_out = df.loc[X_train.index].copy()
    df_train_out[pred_prob_col] = probs_train
    df_train_out[pred_label_col] = preds_train_opt

    df_test_out = df.loc[X_test.index].copy()
    df_test_out[pred_prob_col] = probs_test
    df_test_out[pred_label_col] = preds_test_opt

    front_cols = [col for col in [id_col, target_col, pred_prob_col, pred_label_col] if col in df_train_out.columns]
    other_cols = [col for col in df_train_out.columns if col not in front_cols]
    
    df_train_out = df_train_out[front_cols + other_cols]
    df_test_out = df_test_out[front_cols + other_cols]

    # TRAIN/TEST METRICS
    def calculate_metrics(y_true, probs, preds, dataset_name):
        return {
            'Dataset': dataset_name,
            'Accuracy': accuracy_score(y_true, preds),
            'Precision': precision_score(y_true, preds, zero_division=0),
            'Recall': recall_score(y_true, preds, zero_division=0),
            'F1-Score (Opt)': f1_score(y_true, preds, zero_division=0),
            'AUC-ROC': roc_auc_score(y_true, probs),
            'AU-PR': average_precision_score(y_true, probs)
        }

    metrics_df = pd.DataFrame([
        calculate_metrics(y_train, probs_train, preds_train_opt, 'Train'),
        calculate_metrics(y_test, probs_test, preds_test_opt, 'Test')
    ]).set_index('Dataset')

    print("\n--- Final Performance Summary (Propensity Model - Decision Tree) ---")
    print(metrics_df.round(4))

    # EVALUATION METRICS (ROC, PR, CONFUSION MATRIX)
    fig_eval, axes_eval = plt.subplots(1, 3, figsize=(18, 5))

    # ROC CURVE
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    auc_roc = roc_auc_score(y_test, probs_test)
    
    axes_eval[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_roc:.4f})')
    axes_eval[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    axes_eval[0].set_xlim([0.0, 1.0])
    axes_eval[0].set_ylim([0.0, 1.05])
    axes_eval[0].set_xlabel('False Positive Rate')
    axes_eval[0].set_ylabel('True Positive Rate')
    axes_eval[0].set_title('ROC Curve - Decision Tree (Test Set)', fontsize=12, fontweight='bold')
    axes_eval[0].legend(loc="lower right")
    axes_eval[0].grid(True, linestyle='--', alpha=0.6)

    # PRECISION-RECALL CURVE
    prec_test, rec_test, _ = precision_recall_curve(y_test, probs_test)
    au_pr = average_precision_score(y_test, probs_test)
    
    axes_eval[1].plot(rec_test, prec_test, color='blue', lw=2, label=f'PR (AU-PR = {au_pr:.4f})')
    prec_opt = precision_score(y_test, preds_test_opt, zero_division=0)
    rec_opt = recall_score(y_test, preds_test_opt, zero_division=0)
    axes_eval[1].plot(rec_opt, prec_opt, 'ro', markersize=8, label=f'Threshold ({best_threshold:.3f})')
    
    axes_eval[1].set_xlim([0.0, 1.0])
    axes_eval[1].set_ylim([0.0, 1.05])
    axes_eval[1].set_xlabel('Recall (Sensitivity)')
    axes_eval[1].set_ylabel('Precision')
    axes_eval[1].set_title('Precision-Recall Curve - Decision Tree (Test Set)', fontsize=12, fontweight='bold')
    axes_eval[1].legend(loc="lower left")
    axes_eval[1].grid(True, linestyle='--', alpha=0.6)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, preds_test_opt)
    cm_norm = confusion_matrix(y_test, preds_test_opt, normalize='true')
    
    labels = np.array([[f"{val:,}\n({norm:.1%})" for val, norm in zip(row_cm, row_norm)]
                       for row_cm, row_norm in zip(cm, cm_norm)])

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=axes_eval[2],
                xticklabels=['No Propensity (0)', 'Propensity (1)'],
                yticklabels=['No Propensity (0)', 'Propensity (1)'])
    axes_eval[2].set_xlabel('Model Prediction')
    axes_eval[2].set_ylabel('True Value')
    axes_eval[2].set_title(f'Confusion Matrix - Decision Tree (Test Set)\nThreshold: {best_threshold:.4f}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('dt_evaluation_metrics_summary.png', dpi=150)
    plt.show()
    plt.close()
    print("\n")

    # ==============================================================================
    # VISUALIZACIÓN 2: DECISION TREE DIAGRAM
    # ==============================================================================
    plt.figure(figsize=(22, 10), dpi=300)
    plot_tree(
        best_model,
        max_depth=max_depth_plot,
        feature_names=selected_features,
        class_names=['No Propensity (0)', 'Propensity (1)'],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        precision=2,
        fontsize=8
    )
    title_depth = f"First {max_depth_plot} Levels" if max_depth_plot is not None else "Full Tree"
    plt.title(f'Decision Tree Structure ({title_depth})', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('decision_tree_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("\n")

    # ==============================================================================
    # VISUALIZACIÓN 3: NATIVE FEATURE IMPORTANCE
    # ==============================================================================
    feat_imp = pd.Series(best_model.feature_importances_, index=selected_features, name='Gini Importance')
    feat_imp = feat_imp.sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    feat_imp.tail(15).plot(kind='barh', color='skyblue')
    plt.title('Decision Tree Feature Importance (Gini)', fontsize=12, fontweight='bold')
    plt.xlabel('Gini Importance')
    plt.tight_layout()
    plt.savefig('dt_native_feature_importance.png', dpi=150)
    plt.show()
    plt.close()
    print("\n")

    # ==============================================================================
    # VISUALIZACIÓN 4 & 5: SHAP PLOTS
    # ==============================================================================
    X_test_sel = pd.DataFrame(
        best_selector.transform(X_test),
        columns=selected_features,
        index=X_test.index
    )

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_test_sel)

    if len(shap_values.shape) == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values

    # SHAP Beeswarm
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_to_plot, show=False)
    plt.title('Decision Tree - SHAP Summary Plot', fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('dt_shap_summary_plot.png', dpi=150)
    plt.show()
    plt.close()
    print("\n")
    
    # SHAP Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_to_plot, show=False)
    plt.title('Decision Tree - SHAP Feature Importance (Bar Plot)', fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('dt_shap_bar_importance.png', dpi=150)
    plt.show()
    plt.close()

    return {
        'best_pipeline': best_pipeline,
        'best_model': best_model,
        'best_params': cleaned_params,
        'cv_f1_score_mean': cv_f1_mean,
        'cv_f1_score_std': cv_f1_std,
        'optuna_study': study,
        'selected_features': selected_features,
        'removed_features': removed_features,
        'optimal_threshold': best_threshold,
        'metrics': metrics_df,
        'df_train_results': df_train_out,
        'df_test_results': df_test_out,
        'feature_importance': feat_imp,
        'shap_values': shap_values,
        'explainer': explainer
    }

# COMMAND ----------

# DBTITLE 1,Decision Tree results
results_dt = fit_dt_churn_pipeline(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    df=df,
    target_col='churn',
    id_col='customer_id',
    n_iter_search=20
)

df_test_predictions = results_dt['df_test_results']

target_col = 'churn'
pred_prob_col = f'{target_col}_pred_prob'
pred_label_col = f'{target_col}_pred_label_optimal'

display(df_test_predictions[['customer_id', target_col, pred_prob_col, pred_label_col]].head(20))

# COMMAND ----------

# DBTITLE 1,Random Forest section
# MAGIC %md
# MAGIC ## Logistic Regression (ElasticNet)
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Regularized Logistic Regression - Elastic Net pipeline
def fit_logreg_churn_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    df: pd.DataFrame,
    target_col: str = 'churn',
    id_col: str = 'customer_id',
    class_weight: str | dict | None = 'balanced',
    threshold_fs: str = 'median',
    n_iter_search: int = 20,
    cv_splits: int = 5,
    random_state: int = 42
) -> dict:

    print(" 1. DATASET INFORMATION AND IMBALANCE RATIO")
    print("=" * 60)
    print(f"- Model Algorithm: Logistic Regression (ElasticNet)")
    print(f"- Train set: {X_train.shape[0]} records | {X_train.shape[1]} features")
    print(f"- Test set:  {X_test.shape[0]} records | {X_test.shape[1]} features")

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    imbalance_ratio = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"- Class distribution in Train -> Negative (0): {num_neg:,} | Positive (1): {num_pos:,}")
    print(f"- Calculated imbalance ratio (neg/pos): {imbalance_ratio:.2f}:1")
    print(f"- Logistic Regression Class weight strategy: Optimized by Optuna")

    # HYPERPARAMETER TUNING WITH OPTUNA
    print("\n")
    print(" 2. OPTIMIZATION AND CROSS-VALIDATION (OPTUNA)")
    print("=" * 60)
    
    # CROSS VALIDATION STRATEGY
    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # FEATURE SELECTION ONCE (BEFORE OPTUNA)
    print("\n--- Pre-selecting features (one-time, before optimization) ---")
    fs_model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        C=1.0,
        l1_ratio=0.5,
        max_iter=500,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    fs_selector = SelectFromModel(
        estimator=fs_model,
        threshold=threshold_fs
    )
    
    fs_selector.fit(X_train, y_train)
    
    selected_mask = fs_selector.get_support()
    all_feature_names = X_train.columns.tolist()
    selected_features = [f for f, s in zip(all_feature_names, selected_mask) if s]
    removed_features = [f for f, s in zip(all_feature_names, selected_mask) if not s]
    
    print(f" - Features selected: {len(selected_features)}/{len(all_feature_names)}")
    
    # TRANSFORM ONCE
    X_train_sel = pd.DataFrame(fs_selector.transform(X_train), columns=selected_features, index=X_train.index)
    X_test_sel = pd.DataFrame(fs_selector.transform(X_test), columns=selected_features, index=X_test.index)
    
    # OPTUNA OBJECTIVE FUNCTION (NO FEATURE SELECTION INSIDE)
    def objective(trial):
        params = {
            'C': trial.suggest_categorical('C', [0.01, 0.1, 1.0, 10.0]),
            'l1_ratio': trial.suggest_categorical('l1_ratio', [0.0, 0.5, 1.0]),
            'max_iter': trial.suggest_categorical('max_iter', [500, 1000]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
        
        # CREATE MODEL (NO PIPELINE)
        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            random_state=random_state,
            n_jobs=-1,
            **params
        )
        
        # MANUAL CV LOOP FOR PRUNING (ON PRE-SELECTED FEATURES)
        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train_sel, y_train)):
            X_fold_train, X_fold_val = X_train_sel.iloc[train_idx], X_train_sel.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            fold_score = f1_score(y_fold_val, model.predict(X_fold_val), zero_division=0)
            cv_scores.append(fold_score)
            
            trial.report(np.mean(cv_scores), fold_idx)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(cv_scores)
    
    # CREATE OPTUNA STUDY AND OPTIMIZE
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='logistic_regression_propensity_optimization'
    )
    
    study.optimize(objective, n_trials=n_iter_search, show_progress_bar=True)
    
    # EXTRACT BEST HYPERPARAMETERS
    best_params = study.best_params
    cv_f1_mean = study.best_value
    best_trial = study.best_trial
    cv_f1_std = 0.0
    
    # PRUNING STATISTICS
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print("\n--- Cross-Validation Results (F1-Score Metric) ---")
    print(f" - Best CV F1-Score: {cv_f1_mean:.4f}")
    print(f" - Best Trial Number: {best_trial.number}")
    print(f" - Total Trials Started: {len(study.trials)}")
    print(f" - Trials Completed: {len(completed_trials)}")
    print(f" - Trials Pruned (Early Stopping): {len(pruned_trials)}")
    if len(pruned_trials) > 0:
        print(f" - Time Saved by Pruning: ~{len(pruned_trials) * 15}s (estimated)")
    
    print("\n--- Optimal Hyperparameters Found ---")
    cleaned_params = best_params.copy()
    for param, value in cleaned_params.items():
        print(f" - {param}: {value}")
    
    # TRAIN FINAL MODEL WITH BEST HYPERPARAMETERS (ON PRE-SELECTED FEATURES)
    best_model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        random_state=random_state,
        n_jobs=-1,
        **best_params
    )
    
    best_model.fit(X_train_sel, y_train)

    print("\n")
    print(" 3. FEATURE SELECTION (LOGISTIC REGRESSION)")
    print("=" * 60)
    print(f" - Initial Predictor Features: {len(all_feature_names)}")
    print(f" - Selected Features:          {len(selected_features)}")
    print(f" - Discarded Features:         {len(removed_features)}")

    print("\n--- Kept Features List ---")
    for feat in selected_features:
        print(f"   - {feat}")

    print("\n--- Discarded Features List ---")
    for feat in removed_features:
        print(f"   - {feat}")

    # PROBABILITIES
    probs_train = best_model.predict_proba(X_train_sel)[:, 1]
    probs_test = best_model.predict_proba(X_test_sel)[:, 1]

    # OPTIMAL THRESHOLD
    precisions_tr, recalls_tr, thresholds_tr = precision_recall_curve(y_train, probs_train)
    f1_scores_tr = (2 * precisions_tr * recalls_tr) / (precisions_tr + recalls_tr + 1e-8)
    best_idx = np.argmax(f1_scores_tr)
    best_threshold = thresholds_tr[best_idx] if best_idx < len(thresholds_tr) else 0.5

    print("\n")
    print(" 4. LOGISTIC REGRESSION EVALUATION & OPTIMAL THRESHOLD")
    print("=" * 60)
    print(f"- Optimal Threshold for F1-Score (calculated on Train): {best_threshold:.4f}")

    preds_train_opt = (probs_train >= best_threshold).astype(int)
    preds_test_opt = (probs_test >= best_threshold).astype(int)

    # OUTPUT DATAFRAMES
    pred_prob_col = f'{target_col}_pred_prob'
    pred_label_col = f'{target_col}_pred_label_optimal'
    
    df_train_out = df.loc[X_train.index].copy()
    df_train_out[pred_prob_col] = probs_train
    df_train_out[pred_label_col] = preds_train_opt

    df_test_out = df.loc[X_test.index].copy()
    df_test_out[pred_prob_col] = probs_test
    df_test_out[pred_label_col] = preds_test_opt

    front_cols = [col for col in [id_col, target_col, pred_prob_col, pred_label_col] if col in df_train_out.columns]
    other_cols = [col for col in df_train_out.columns if col not in front_cols]
    
    df_train_out = df_train_out[front_cols + other_cols]
    df_test_out = df_test_out[front_cols + other_cols]

    # TRAIN/TEST METRICS
    def calculate_metrics(y_true, probs, preds, dataset_name):
        return {
            'Dataset': dataset_name,
            'Accuracy': accuracy_score(y_true, preds),
            'Precision': precision_score(y_true, preds, zero_division=0),
            'Recall': recall_score(y_true, preds, zero_division=0),
            'F1-Score (Opt)': f1_score(y_true, preds, zero_division=0),
            'AUC-ROC': roc_auc_score(y_true, probs),
            'AU-PR': average_precision_score(y_true, probs)
        }

    metrics_df = pd.DataFrame([
        calculate_metrics(y_train, probs_train, preds_train_opt, 'Train'),
        calculate_metrics(y_test, probs_test, preds_test_opt, 'Test')
    ]).set_index('Dataset')

    print("\n--- Final Performance Summary (Propensity Model - Logistic Regression) ---")
    print(metrics_df.round(4))

    # EVALUATION METRICS (ROC, PR, CONFUSION MATRIX)
    fig_eval, axes_eval = plt.subplots(1, 3, figsize=(18, 5))

    # ROC CURVE
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    auc_roc = roc_auc_score(y_test, probs_test)
    
    axes_eval[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_roc:.4f})')
    axes_eval[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    axes_eval[0].set_xlim([0.0, 1.0])
    axes_eval[0].set_ylim([0.0, 1.05])
    axes_eval[0].set_xlabel('False Positive Rate')
    axes_eval[0].set_ylabel('True Positive Rate')
    axes_eval[0].set_title('ROC Curve - Logistic Regression (Test Set)', fontsize=12, fontweight='bold')
    axes_eval[0].legend(loc="lower right")
    axes_eval[0].grid(True, linestyle='--', alpha=0.6)

    # PRECISION-RECALL CURVE
    prec_test, rec_test, _ = precision_recall_curve(y_test, probs_test)
    au_pr = average_precision_score(y_test, probs_test)
    
    axes_eval[1].plot(rec_test, prec_test, color='blue', lw=2, label=f'PR (AU-PR = {au_pr:.4f})')
    prec_opt = precision_score(y_test, preds_test_opt, zero_division=0)
    rec_opt = recall_score(y_test, preds_test_opt, zero_division=0)
    axes_eval[1].plot(rec_opt, prec_opt, 'ro', markersize=8, label=f'Threshold ({best_threshold:.3f})')
    
    axes_eval[1].set_xlim([0.0, 1.0])
    axes_eval[1].set_ylim([0.0, 1.05])
    axes_eval[1].set_xlabel('Recall (Sensitivity)')
    axes_eval[1].set_ylabel('Precision')
    axes_eval[1].set_title('Precision-Recall Curve - Logistic Regression (Test Set)', fontsize=12, fontweight='bold')
    axes_eval[1].legend(loc="lower left")
    axes_eval[1].grid(True, linestyle='--', alpha=0.6)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, preds_test_opt)
    cm_norm = confusion_matrix(y_test, preds_test_opt, normalize='true')
    
    labels = np.array([[f"{val:,}\n({norm:.1%})" for val, norm in zip(row_cm, row_norm)]
                       for row_cm, row_norm in zip(cm, cm_norm)])

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=axes_eval[2],
                xticklabels=['No Propensity (0)', 'Propensity (1)'],
                yticklabels=['No Propensity (0)', 'Propensity (1)'])
    axes_eval[2].set_xlabel('Model Prediction')
    axes_eval[2].set_ylabel('True Value')
    axes_eval[2].set_title(f'Confusion Matrix - Logistic Regression (Test Set)\nThreshold: {best_threshold:.4f}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('logreg_evaluation_metrics_summary.png', dpi=150)
    plt.show()
    plt.close()
    print("\n")

    # FEATURE COEFFICIENTS (LOGISTIC REGRESSION)
    coefficients = pd.Series(best_model.coef_[0], index=selected_features, name='Coefficient')
    coefficients_abs = coefficients.abs().sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    coefficients_abs.tail(15).plot(kind='barh', color='steelblue')
    plt.title('Logistic Regression Feature Importance (Absolute Coefficients)', fontsize=12, fontweight='bold')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig('logreg_feature_coefficients.png', dpi=150)
    plt.show()
    plt.close()
    print("\n")

    return {
        'best_pipeline': None,  # No pipeline used (feature selection done once)
        'best_model': best_model,
        'best_params': cleaned_params,
        'cv_f1_score_mean': cv_f1_mean,
        'cv_f1_score_std': cv_f1_std,
        'optuna_study': study,
        'selected_features': selected_features,
        'removed_features': removed_features,
        'optimal_threshold': best_threshold,
        'metrics': metrics_df,
        'df_train_results': df_train_out,
        'df_test_results': df_test_out,
        'coefficients': coefficients,
        'coefficients_abs': coefficients_abs
    }

# COMMAND ----------

# DBTITLE 1,Regularized Logistic Regression - Elastic Net
results_logreg = fit_logreg_churn_pipeline(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    df=df,
    target_col='churn',
    id_col='customer_id',
    n_iter_search=15
)

df_test_predictions = results_logreg['df_test_results']

target_col = 'churn'
pred_prob_col = f'{target_col}_pred_prob'
pred_label_col = f'{target_col}_pred_label_optimal'

display(df_test_predictions[['customer_id', target_col, pred_prob_col, pred_label_col]].head(20))

# COMMAND ----------

# DBTITLE 1,Comparación final de modelos
# MAGIC %md
# MAGIC ---
# MAGIC ## Final Model Comparison and Recommendation
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Model comparison function
def compare_models_and_recommend(
    model_results: dict,
    total_features: int = 16
) -> pd.DataFrame:
    
    # PREPARE COMPARISON DATA FROM ALL MODELS
    comparison_data = []
    
    for model_name, results in model_results.items():
        test_metrics = results['metrics'].loc['Test']
        comparison_data.append({
            'Model': model_name,
            'Test F1': test_metrics['F1-Score (Opt)'],
            'Test AUC-ROC': test_metrics['AUC-ROC'],
            'Test Precision': test_metrics['Precision'],
            'Test Recall': test_metrics['Recall'],
            'Test Accuracy': test_metrics['Accuracy'],
            'Test AU-PR': test_metrics['AU-PR'],
            'CV F1': results['cv_f1_score_mean'],
            'Features': len(results['selected_features']),
            'Threshold': results['optimal_threshold'],
            'Best Params': results['best_params']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Test F1', ascending=False).reset_index(drop=True)
    
    print(" " + "="*78 + " ")
    print("MODEL COMPARISON - TEST SET PERFORMANCE")
    print(" " + "="*78 + " ")
    print()
    print(df_comparison.round(4).to_string(index=False))
    print()
    print(" " + "="*78 + " ")
    
    # VISUAL COMPARISON PLOTS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1 VS AUC SCATTER PLOT
    ax1 = axes[0]
    for idx, row in df_comparison.iterrows():
        ax1.scatter(row['Test AUC-ROC'], row['Test F1'], s=200, alpha=0.7, label=row['Model'])
        ax1.annotate(row['Model'], 
                     (row['Test AUC-ROC'], row['Test F1']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('AUC-ROC (Test)', fontweight='bold')
    ax1.set_ylabel('F1-Score (Test)', fontweight='bold')
    ax1.set_title('Performance Trade-off: F1 vs AUC-ROC', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.900, 0.912)
    ax1.set_ylim(0.850, 0.856)
    
    # BAR CHART COMPARISON
    ax2 = axes[1]
    metrics_to_plot = ['Test F1', 'Test Precision', 'Test Recall']
    df_plot = df_comparison.set_index('Model')[metrics_to_plot]
    df_plot.plot(kind='bar', ax=ax2, width=0.75, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Key Metrics Comparison', fontsize=12, fontweight='bold')
    ax2.legend(title='Metric', loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_ylim(0.80, 0.90)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('model_comparison_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    print()
    
    # PRINT DIRECT RECOMMENDATION
    print(" " + "="*78 + " ")
    print("RECOMMENDATION")
    print(" " + "="*78 + " ")
    print()
    
    top_model = df_comparison.iloc[0]
    print(f"Go with: {top_model['Model']}")
    print()
    print("Stats:")
    print(f"  • Test F1: {top_model['Test F1']:.4f}")
    print(f"  • Test Accuracy: {top_model['Test Accuracy']:.4f}")
    print(f"  • AUC-ROC: {top_model['Test AUC-ROC']:.4f}")
    print(f"  • AU-PR: {top_model['Test AU-PR']:.4f}")
    print(f"  • Precision: {top_model['Test Precision']:.4f}")
    print(f"  • Recall: {top_model['Test Recall']:.4f}")
    print(f"  • Selected features: {top_model['Features']}/{total_features}")
    print(f"  • Optimal threshold: {top_model['Threshold']:.4f}")
    print()
    print("Optimal Hyperparameters:")
    for param, value in top_model['Best Params'].items():
        print(f"  • {param}: {value}")
    print(" " + "="*78 + " ")
    
    return df_comparison

# COMMAND ----------

# DBTITLE 1,Summary
# Prepare model results dictionary
model_results = {
    'XGBoost': results_xgboost,
    'LightGBM': results_lgbm,
    'Decision Tree': results_dt,
    'Logistic Regression': results_logreg
}

# Generate comparison 
df_comparison = compare_models_and_recommend(
    model_results=model_results,
    total_features=16
)