import unittest
from data.data_loader import load_data

class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        day_df, hour_df = load_data('data/day.csv', 'data/hour.csv')
        self.assertFalse(day_df.empty)
        self.assertFalse(hour_df.empty)
        self.assertIn('dteday', day_df.columns)
        self.assertIn('dteday', hour_df.columns)

