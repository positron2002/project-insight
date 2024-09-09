import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data(df):
    """Preprocesses data for machine learning or visualization tasks.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    # Handle missing values in numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')
    print(f"Shape of numeric columns before imputation: {df[numeric_columns].shape}")
    numeric_data = numeric_imputer.fit_transform(df[numeric_columns])
    print(f"Shape of numeric data after imputation: {numeric_data.shape}")
    df[numeric_columns] = pd.DataFrame(numeric_data, columns=numeric_columns, index=df.index)

    # Handle missing values and categorical encoding for non-numeric columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    non_numeric_data = categorical_imputer.fit_transform(df[non_numeric_columns])

    # One-hot encode categorical features (consider other encoding methods if needed)
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(non_numeric_data)
    encoded_columns = [f"{col}_{cat}" for col, cats in zip(non_numeric_columns, encoder.categories_) for cat in cats]

    # Combine numeric and encoded categorical data into a single DataFrame
    preprocessed_df = pd.concat([df[numeric_columns], pd.DataFrame(encoded_data, columns=encoded_columns)], axis=1)

    # Standardize numeric features (optional)
    scaler = StandardScaler()
    scaled_numeric_data = scaler.fit_transform(preprocessed_df[numeric_columns])
    print(f"Shape of numeric data after scaling: {scaled_numeric_data.shape}")
    preprocessed_df[numeric_columns] = pd.DataFrame(scaled_numeric_data, columns=numeric_columns, index=df.index)

    return preprocessed_df
