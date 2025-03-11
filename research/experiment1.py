import os
os.environ["DAGSHUB_DISABLE_SSL_VERIFY"] = "true"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import dagshub

dagshub.init(repo_owner='ay747283', repo_name='ImagoAI-Assignment', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/ay747283/ImagoAI-Assignment.mlflow')
mlflow.set_experiment('Regressor models')

df = pd.read_csv("data/TASK-ML-INTERN.csv") 

X = df.iloc[:, 1:21]  # Using columns 1 to 20 as features
y = df.iloc[:, -1]    # Using the last column as the target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoostRegressor": AdaBoostRegressor(n_estimators=100, random_state=42),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "SVR": SVR(),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5)
}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("Test_Size", 0.2)
        mlflow.log_param("Random_State", 42)
        mlflow.log_metric("Mean_Squared_Error", mse)
        mlflow.log_metric("Mean_Absolute_Error", mae)
        mlflow.log_metric("R2_Score", r2)
        
        mlflow.sklearn.log_model(model, f"{model_name}_Model")

        print(f"{model_name}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")



## It suggest that we have to use GradientBoostingRegressor as it has the lowest MSE and MAE and highest R² score