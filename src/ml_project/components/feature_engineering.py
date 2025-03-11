import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")


def standard_scaling(df):
    scaler = StandardScaler()
    numeric_df = df.select_dtypes(include=[np.number])
    scaled_array = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_df.columns, index=df.index)
    return scaled_df

def data_preprocessing(train_data, test_data):
    train_data = standard_scaling(train_data)
    test_data = standard_scaling(test_data)
    return train_data, test_data

train_preprocessed_data, test_preprocessed_data = data_preprocessing(train_data, test_data)

data_path = os.path.join("data", "features")
os.makedirs(data_path, exist_ok=True)

train_preprocessed_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_preprocessed_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
