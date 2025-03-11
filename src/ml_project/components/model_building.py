import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingRegressor


train_data = pd.read_csv("data/features/train.csv")
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

## on the basis of the experiment 3, we have these parameters as the best parameters
params = {'learning_rate': 0.01, 
        'max_depth': 3, 
        'max_features': 0.5, 
        'n_estimators': 50, 
        'subsample': 0.5
}

clf = GradientBoostingRegressor(**params , random_state=42)
clf.fit(X_train, y_train)

## save the model
pickle.dump(clf , open('model.pkl' , 'wb'))