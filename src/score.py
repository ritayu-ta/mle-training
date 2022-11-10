import os
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import configure_logger


PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
score_logger = configure_logger(log_file=os.path.join(PARENT_PATH, "logs/score.log"))

score_logger.info("Model Scoring - Started")
# user argument for output data path
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Path for model file")
parser.add_argument("--dataset", help="Path for scoring data file")
parser.add_argument("--output", help="Path for saving any output files")
args = parser.parse_args()

if args.model:
    MODEL_PATH = PARENT_PATH + "/" + args.model + "/"
else:
    MODEL_PATH = PARENT_PATH + "/artifacts/"

if args.dataset:
    DATA_PATH = PARENT_PATH + "/" + args.dataset + "/"
else:
    DATA_PATH = PARENT_PATH + "/data/processed/"

if args.output:
    OUTPUT_PATH = PARENT_PATH + "/" + args.output + "/"
else:
    OUTPUT_PATH = PARENT_PATH + "/results/"


# loading files
X_test_prepared = pd.read_csv(DATA_PATH + "X_test_prepared.csv")
y_test = pd.read_csv(DATA_PATH + "y_test.csv")
final_model = pickle.load(open(MODEL_PATH + "final_model.sav", "rb"))
score_logger.info("Model Scoring - Data Loading - Completed")

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
results = pd.DataFrame(list(zip([final_mse], [final_rmse])), columns=["MSE", "RMSE"])
score_logger.info("Model Scoring - Predictions - Generated")

# writing scoring results
results.to_csv(OUTPUT_PATH + "scoring.csv", index=False)
score_logger.info("Model Scoring - Results - Saved")
score_logger.info("Model Scoring - Completed")
