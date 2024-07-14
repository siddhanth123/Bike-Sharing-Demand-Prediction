import pandas as pd

def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Summary DataFrame.
    """
    summary = pd.DataFrame({
        'Feature': df.columns.values,
        'Datatype': df.dtypes.values,
        'Negative_Values': [1 if (df[col].values < 0).any() else 0 for col in df.select_dtypes(include='number').columns],
        'Null_Values': df.isna().sum(),
        'Unique_value_count': df.nunique().values,
        'Unique_values': [df[col].unique() for col in df.columns],
        'Duplicate Rows': df.duplicated().sum()
    }).reset_index(drop=True)
    
    return summary