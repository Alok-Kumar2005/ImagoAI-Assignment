import numpy as np
import pandas as pd
import os
import logging
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


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from the given path.
    :param data_path: Path to the data file.
    :return: DataFrame containing the data.
    """
    try:
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        logger.info("Data loaded successfully")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found at {data_path}: {e}")
        raise FileNotFoundError(f"File not found at {data_path}") from e


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by removing the first column.
    :param df: DataFrame containing the data.
    :return: DataFrame containing the preprocessed data.
    """
    try:
        # Remove the first column (assumed to be an index or ID column)
        final_df = df.iloc[:, 1:]
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
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)
        save_data(train_data, test_data, "data/raw")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise e


if __name__ == "__main__":
    main()
