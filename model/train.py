import time
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb


def cross_val_evaluate(model, X, y, cv=5) -> dict:
    """
    Evaluate model performance with cross-validation.

    Args:
        model: Scikit-learn compatible model.
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        cv (int): Number of cross-validation folds.

    Returns:
        dict: Cross-validated metrics.
    """
    start_time = time.time()
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error',
        'R2': 'r2',
        'MAPE': 'neg_mean_absolute_percentage_error'
    }

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
    training_time = time.time() - start_time
    
    metrics = {
        'MAE': -scores['test_MAE'].mean(),
        'MSE': -scores['test_MSE'].mean(),
        'RMSE': np.sqrt(-scores['test_MSE'].mean()),
        'R2': scores['test_R2'].mean(),
        'MAPE': -scores['test_MAPE'].mean(),
        'Training Time': training_time
    }
    
    return metrics


def train_models(X_train, y_train) -> pd.DataFrame:
    """
    Train various models and evaluate their performance.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        pd.DataFrame: Performance metrics for each model.
    """
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(),
        'DecisionTree': DecisionTreeRegressor(),
        'LightGBM': lgb.LGBMRegressor(),
        'XGBoost': xgb.XGBRegressor(),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        metrics = cross_val_evaluate(model, X_train, y_train.values.ravel())
        results[name] = metrics
    
    return pd.DataFrame(results).T