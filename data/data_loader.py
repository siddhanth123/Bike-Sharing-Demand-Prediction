import pandas as pd
from typing import Tuple

def load_data(day_path: str, hour_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the bike sharing datasets.

    Args:
        day_path (str): Path to the day dataset CSV file.
        hour_path (str): Path to the hour dataset CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for day and hour datasets.
    """
    day_df = pd.read_csv(day_path, parse_dates=['dteday'])
    hour_df = pd.read_csv(hour_path, parse_dates=['dteday'])

    day_df['dteday'] = day_df['dteday'].dt.tz_localize(None)
    hour_df['dteday'] = hour_df['dteday'].dt.tz_localize(None)
    
    return day_df, hour_df