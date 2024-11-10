# data_preprocessing.py
import pandas as pd

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Simple preprocessing steps: fill missing values, encode categories."""
    data.fillna(data.mean(), inplace=True)
    # Add more preprocessing steps as needed
    return data
