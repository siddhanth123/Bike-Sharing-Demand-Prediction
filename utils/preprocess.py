from typing import Tuple
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def preprocess_data(hour_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess the hour dataset.

    Args:
        hour_df (pd.DataFrame): Input hour DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Preprocessed train/test data and labels.
    """
    transformed_df = hour_df.copy(deep=False)

    # winsorize the features
    transformed_df['hum'] = winsorize(transformed_df['hum'], limits=(0.01, 0.1))
    transformed_df['windspeed'] = winsorize(transformed_df['windspeed'], limits=(0.01, 0.1))

    # Cyclical encoding for hour
    transformed_df['hour_sin'] = np.sin(2 * np.pi * transformed_df['hr'] / 24)
    transformed_df['hour_cos'] = np.cos(2 * np.pi * transformed_df['hr'] / 24)

    # Dropping certain features that cause multicollinearity and redundancy that negatively affect model performance
    transformed_df = transformed_df.drop(labels=['dteday','atemp','casual','registered', 'year'], axis=1)

    # Seprating Independent and dependent features
    X = transformed_df.drop(labels=['cnt'], axis=1)
    y = transformed_df[['cnt']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

    # Define numerical and categorical columns
    numerical_cols = ['temp', 'hum', 'windspeed', 'hour_sin', 'hour_cos']
    binary_cols = ['holiday', 'workingday', 'yr', 'day']
    one_hot_cols = ['season', 'mnth', 'weekday', 'weathersit']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    one_hot_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine the pipelines into the ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('onehot', one_hot_pipeline, one_hot_cols),
        ('binary', 'passthrough', binary_cols)
    ])

    # Initialize the scaler for the target variable
    target_scaler = StandardScaler()

    # Fit the scaler on the training target and transform
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))

    # Transform the testing target
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

    # # Fit the preprocessor on the data
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # convert the preprocessed data back to a DataFrame
    # Get the feature names after preprocessing
    one_hot_feature_names = preprocessor.named_transformers_['onehot']['onehot'].get_feature_names_out(one_hot_cols)
    binary_feature_names = binary_cols  # Directly passed through features
    num_feature_names = numerical_cols  # Scaled numerical features

    # Combine all feature names
    all_feature_names = np.concatenate([num_feature_names, one_hot_feature_names, binary_feature_names])
    # Debugging: Print the number of feature names

    # Create DataFrame with preprocessed data
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names)

    return X_train_processed_df, X_test_processed_df, y_train, y_test