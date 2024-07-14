import unittest
import pandas as pd
import numpy as np
from model.tune import tune_hyperparameters

class TestTune(unittest.TestCase):

    def setUp(self):
        # Setup a simple dataset for testing
        self.X_train = pd.DataFrame(np.random.rand(100, 5), columns=list('ABCDE'))
        self.y_train = pd.Series(np.random.rand(100))

    def test_tune_hyperparameters(self):
        best_models = tune_hyperparameters(self.X_train, self.y_train)
        
        self.assertIn('RandomForest', best_models)
        self.assertIn('XGBoost', best_models)
        self.assertIsNotNone(best_models['RandomForest'])
        self.assertIsNotNone(best_models['XGBoost'])
