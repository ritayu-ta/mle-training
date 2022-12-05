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


PARENT_PATH = os.path.dirname(os.path.dirname(__file__))

with open(PARENT_PATH + "/deploy/logger/config.json") as jsonfile:
    # `json.loads` parses a string in json format
    LOGGING_DEFAULT_CONFIG = json.load(jsonfile)["config"]

print(LOGGING_DEFAULT_CONFIG)
