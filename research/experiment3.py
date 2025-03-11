import os
import mlflow 
import mlflow.sklearn
import dagshub
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_feature_train = pd.read_csv("data/features/train.csv")
X_feature_test = pd.read_csv("data/features/test.csv")

X_train = X_feature_train.iloc[:, :-1]
y_train = X_feature_train.iloc[:, -1]

X_test = X_feature_test.iloc[:, :-1]
y_test = X_feature_test.iloc[:, -1]

model = GradientBoostingRegressor()
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 1],
    "subsample": [0.5, 0.7, 1.0],
    "max_features": [0.5, 0.7, 1.0]
}
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)

dagshub.init(repo_owner='ay747283', repo_name='ImagoAI-Assignment', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/ay747283/ImagoAI-Assignment.mlflow')
mlflow.set_experiment('GBR Hyperparameter Tuning')

with mlflow.start_run() as parent_run:
    grid_search.fit(X_train, y_train)

    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("mean_test_score", grid_search.cv_results_["mean_test_score"][i])

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("best_mean_test_score", best_score)

    train_df = X_train.copy()
    train_df['target'] = y_train
    train_csv_path = "train_data.csv"
    train_df.to_csv(train_csv_path, index=False)
    mlflow.log_artifact(train_csv_path, artifact_path="training_data")

    test_df = X_test.copy()
    test_df['target'] = y_test
    test_csv_path = "test_data.csv"
    test_df.to_csv(test_csv_path, index=False)
    mlflow.log_artifact(test_csv_path, artifact_path="testing_data")

    script_file = __file__ if '__file__' in globals() else "script.py"
    mlflow.log_artifact(script_file)

    mlflow.sklearn.log_model(grid_search.best_estimator_, "Gradient_Boosting_Regressor")

    print("Best Parameters:", best_params)
    print("Best Mean Test Score:", best_score)
