import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score

clf = pickle.load(open('model.pkl', 'rb'))
test_data = pd.read_csv("data/features/test.csv")
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

y_pred = clf.predict(X_test)
mean_squared_error_value = mean_squared_error(y_test, y_pred)
mean_absolute_error_value = mean_absolute_error(y_test, y_pred)
r2_score_value = r2_score(y_test, y_pred)


metrics_dict = {
    'mean_squared_error': mean_squared_error_value,
    'mean_absolute_error': mean_absolute_error_value,
    'r2_score': r2_score_value
}

with open('metrics.json', 'w') as outfile:
    json.dump(metrics_dict, outfile)
