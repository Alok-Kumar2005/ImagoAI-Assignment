import numpy as np
import pandas as pd
import os
import logging
import yaml
from sklearn.model_selection import train_test_split

# Set up logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """"Load parameters from the given path."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded successfully from {params_path}")
        return params
    except FileNotFoundError as e:
        logger.error(f"File not found at {params_path}: {e}")
        raise FileNotFoundError(f"File not found at {params_path}") from e
    except Exception as e:
        logger.error(f"Error in loading parameters: {e}")
        raise e



def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from the given path.
    :param data_path: Path to the data file.
    :return: DataFrame containing the data.
    """
    try:
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        data = data.iloc[:, 1:]
        logger.info("Data loaded successfully")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found at {data_path}: {e}")
        raise FileNotFoundError(f"File not found at {data_path}") from e


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by keeping only the first 21 columns and the last column (target),
    dropping all columns from index 21 to the second-to-last column.
    
    :param df: DataFrame containing the data.
    :return: DataFrame containing the preprocessed data.
    """
    try:
        # Create a list of column indices to keep: first 20 columns (0 to 19) + last column
        keep_indices = list(range(20)) + [-1]
        final_df = df.iloc[:, keep_indices]
        return final_df
    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        raise e


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    """
    Save the train and test data to the given path.
    :param train_data: DataFrame containing the training data.
    :param test_data: DataFrame containing the testing data.
    :param data_path: Path to save the data.
    """
    try:
        logger.info(f"Saving train and test data to {data_path}")
        os.makedirs(data_path, exist_ok=True)
        train_file_path = os.path.join(data_path, "train.csv")
        test_file_path = os.path.join(data_path, "test.csv")
        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error in saving data: {e}")
        raise e


def main():
    try:
        df = load_data("data/TASK-ML-INTERN.csv")
        final_df = preprocess_data(df)
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, "data/raw")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise e


if __name__ == "__main__":
    main()
