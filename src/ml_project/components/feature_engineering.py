import pandas as pd
import numpy as np
import os
import pickle
import logging
from sklearn.preprocessing import StandardScaler


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

train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")


## using standard scaling to scale the data
def standard_scaling(df):
    scaler = StandardScaler()
    df = df.iloc[:, :-1]
    # numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = df
    scaled_array = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_df.columns, index=df.index)
    pickle.dump(scaler , open('scaler.pkl' , 'wb'))
    return scaled_df

def data_preprocessing(train_data, test_data):
    train_data = standard_scaling(train_data)
    test_data = standard_scaling(test_data)
    return train_data, test_data


## saving the data in the data/features directory
train_preprocessed_data, test_preprocessed_data = data_preprocessing(train_data, test_data)

data_path = os.path.join("data", "features")
os.makedirs(data_path, exist_ok=True)

train_preprocessed_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_preprocessed_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
