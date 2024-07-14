import unittest
import pandas as pd
from utils.preprocess import preprocess_data

class TestPreprocess(unittest.TestCase):

    def test_preprocess_data(self):
        sample_data = {
            'instant': [1, 2, 3],
            'dteday': pd.to_datetime(['2011-01-01', '2011-01-02', '2011-01-03']),
            'season': [1, 1, 1],
            'yr': [0, 0, 0],
            'mnth': [1, 1, 1],
            'hr': [0, 1, 2],
            'holiday': [0, 0, 0],
            'weekday': [6, 0, 1],
            'workingday': [0, 0, 1],
            'weathersit': [1, 1, 1],
            'temp': [0.24, 0.22, 0.22],
            'atemp': [0.2879, 0.2727, 0.2727],
            'hum': [0.81, 0.80, 0.80],
            'windspeed': [0.0, 0.0, 0.0],
            'casual': [3, 8, 5],
            'registered': [13, 32, 27],
            'cnt': [16, 40, 32]
        }
        hour_df = pd.DataFrame(sample_data)
        
        X_train, X_test, y_train, y_test = preprocess_data(hour_df)
        
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

if __name__ == '__main__':
    unittest.main()
