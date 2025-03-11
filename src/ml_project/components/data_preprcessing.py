import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("data/raw/train.csv")
test_data = pd.read_csv("data/raw/test.csv")


## to handle the outliers
def outliers_handling(df):
    for col in df.columns:
        if df[col].dtype != 'object':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


## applying the outliers handling
def data_preprocessing(train_data, test_data):
    train_data = outliers_handling(train_data)
    test_data = outliers_handling(test_data)
    return train_data, test_data

train_preprocessed_data, test_preprocessed_data = data_preprocessing(train_data, test_data)


## saved to the data/processed directory
data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)

train_preprocessed_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_preprocessed_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
