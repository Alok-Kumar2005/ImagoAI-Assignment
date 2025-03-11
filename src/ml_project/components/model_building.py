import numpy as np
import pandas as pd
import pickle
import logging
import yaml

from sklearn.ensemble import GradientBoostingRegressor


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



X_train = pd.read_csv("data/features/train.csv")
print(X_train.shape)

y_train = pd.read_csv("data/processed/train.csv").iloc[:, -1]
# X_train = train_data.iloc[:, :-1]
# y_train = train_data.iloc[:,]

## on the basis of the experiment 3, we have these parameters as the best parameters
# params = {'learning_rate': 0.01, 
#         'max_depth': 3, 
#         'max_features': 0.5, 
#         'n_estimators': 50, 
#         'subsample': 0.5
# }

params = load_params('params.yaml')['model_building']

print(params)

clf = GradientBoostingRegressor(**params , random_state=42)
clf.fit(X_train, y_train)

## save the model
pickle.dump(clf , open('model.pkl' , 'wb'))