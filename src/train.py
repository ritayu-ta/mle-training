import os
import argparse
import json
import pickle


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from utils import configure_logger

PARENT_PATH = os.path.dirname(os.path.dirname(__file__))

train_logger = configure_logger(log_file=os.path.join(PARENT_PATH, "logs/train.log"))

train_logger.info("Model Training - Started")
# user argument for output data path
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path for input data")
parser.add_argument("--output", help="Path for storing modeling files")
args = parser.parse_args()

if args.input:
    DATA_PATH = PARENT_PATH + "/" + args.input + "/"
else:
    DATA_PATH = PARENT_PATH + "/data/processed/"

if args.output:
    MODEL_PATH = PARENT_PATH + "/" + args.output + "/"
else:
    MODEL_PATH = PARENT_PATH + "/artifacts/"

# loading files
housing_prepared = pd.read_csv(DATA_PATH + "housing_prepared.csv")
housing_labels = pd.read_csv(DATA_PATH + "housing_labels.csv")


lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

train_logger.info("Model Training - Linear Regression - Completed")

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
train_logger.info("Model Training - Decision Tree - Completed")

param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}


forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
)
rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
train_logger.info("Model Training - Random Forest - Completed")

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, housing_prepared.columns), reverse=True)


final_model = grid_search.best_estimator_
train_logger.info("Model Training - Hyperparameter Tuning - Completed")

# saving final model
pickle.dump(final_model, open(MODEL_PATH + "final_model.sav", "wb"))
train_logger.info("Model Training - Final Model Saved")
train_logger.info("Model Training - Completed")
