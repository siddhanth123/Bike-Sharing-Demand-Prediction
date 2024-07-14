# Evaluate tuned models on the training set using cross-validation
import time
import numpy as np
import pandas as pd
from sklearn.base import r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from model.train import cross_val_evaluate


def evaluate_tune_models_train_set(best_models, X_train, y_train, results_df, top_models) -> pd.DataFrame:
    tuned_results = {}
    for model_name, model in best_models.items():
        metrics = cross_val_evaluate(model, X_train, y_train.values.ravel())
        tuned_results[model_name] = metrics

    # Display the tuned results
    tuned_results_df = pd.DataFrame(tuned_results).T

    # Comparison table before and after tuning
    comparison_df = results_df.loc[top_models].join(tuned_results_df, lsuffix='_before', rsuffix='_after')

    return comparison_df


# Helper function to evaluate model performance on the test set
def test_set_evaluate(model, X_test, y_test) -> dict:
    start_time = time.time()
    
    y_pred = model.predict(X_test)
    evaluation_time = time.time() - start_time
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'Evaluation Time': evaluation_time
    }
    
    return metrics


# Evaluate the final models on the test set
def evaluate_tune_models_test_set(best_models, X_test, y_test) -> pd.DataFrame:
    test_results = {}
    for model_name, model in best_models.items():
        metrics = test_set_evaluate(model, X_test, y_test.values.ravel())
        test_results[model_name] = metrics

    # Display the test results
    test_results_df = pd.DataFrame(test_results).T
    
    return test_results_df