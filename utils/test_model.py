import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from model.train import cross_val_evaluate, train_models

class TestModel(unittest.TestCase):

    def setUp(self):
        # Setup a simple dataset for testing
        self.X_train = pd.DataFrame(np.random.rand(100, 5), columns=list('ABCDE'))
        self.y_train = pd.Series(np.random.rand(100))

    def test_cross_val_evaluate(self):
        model = Ridge()
        metrics = cross_val_evaluate(model, self.X_train, self.y_train)
        
        self.assertIn('MAE', metrics)
        self.assertIn('MSE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        self.assertIn('MAPE', metrics)
        self.assertIn('Training Time', metrics)

    def test_train_models(self):
        results = train_models(self.X_train, self.y_train)
        
        self.assertFalse(results.empty)
        self.assertIn('Ridge', results.index)
        self.assertIn('Lasso', results.index)
        self.assertIn('SVR', results.index)
        self.assertIn('RandomForest', results.index)
        self.assertIn('DecisionTree', results.index)
        self.assertIn('LightGBM', results.index)
        self.assertIn('XGBoost', results.index)

