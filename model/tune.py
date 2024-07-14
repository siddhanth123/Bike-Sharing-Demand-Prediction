import time
import numpy as np
import pandas as pd
from sklearn.base import r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from model.train import cross_val_evaluate

def tune_hyperparameters(top_models, X_train, y_train) -> dict:
    """
    Tune hyperparameters for top models.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        dict: Best models with tuned hyperparameters.
    """
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }

    # Models to evaluate
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(),
        'DecisionTree': DecisionTreeRegressor(),
        'LightGBM': lgb.LGBMRegressor(),
        'XGBoost': xgb.XGBRegressor(),
    #     'CatBoost': CatBoostRegressor(silent=True)
    }
    
    best_models = {}
    for model_name in top_models:
        print(f"Tuning hyperparameters for {model_name}...")
        model = models[model_name]
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train.values.ravel())
        
        best_models[model_name] = grid_search.best_estimator_

    return best_models
