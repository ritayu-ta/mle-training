import argparse
import os
import pickle
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from utils import configure_logger

# path to files in package
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
PARENT_PATH = os.path.dirname(os.path.dirname(__file__))
HOUSING_PATH = os.path.join(PARENT_PATH + "/data/raw", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

ingestion_logger = configure_logger(
    log_file=os.path.join(PARENT_PATH, "logs/ingest_data.log")
)

ingestion_logger.info("Data Ingestion - Started")


# function to pull data from URL
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# function to load housing data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()
ingestion_logger.info("Data Ingestion - Data Download - Completed")

# split data into training and validation
test_split_prop = 0.2
train_set, test_set = train_test_split(
    housing, test_size=test_split_prop, random_state=42
)

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()
compare_props["Rand. %error"] = (
    100 * compare_props["Random"] / compare_props["Overall"] - 100
)
compare_props["Strat. %error"] = (
    100 * compare_props["Stratified"] / compare_props["Overall"] - 100
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude")
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]

housing["bedrooms_per_room"] = (
    housing["total_bedrooms"] / housing["total_rooms"]
)
housing["population_per_household"] = (
    housing["population"] / housing["households"]
)


housing = strat_train_set.drop(
    "median_house_value", axis=1
)  # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
ingestion_logger.info("Data Ingestion - Data Split - Completed")

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
X = imputer.transform(housing_num)
ingestion_logger.info("Data Ingestion - Data Imputation - Completed")

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

housing_tr["rooms_per_household"] = (
    housing_tr["total_rooms"] / housing_tr["households"]
)

housing_tr["bedrooms_per_room"] = (
    housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
)
housing_tr["population_per_household"] = (
    housing_tr["population"] / housing_tr["households"]
)

housing_cat = housing[["ocean_proximity"]]

housing_prepared = housing_tr.join(
    pd.get_dummies(housing_cat, drop_first=True)
)
ingestion_logger.info("Data Ingestion - Trainng Data - Prepared")

# validation data
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(
    X_test_prepared, columns=X_test_num.columns, index=X_test.index
)
X_test_prepared["rooms_per_household"] = (
    X_test_prepared["total_rooms"] / X_test_prepared["households"]
)
X_test_prepared["bedrooms_per_room"] = (
    X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
)
X_test_prepared["population_per_household"] = (
    X_test_prepared["population"] / X_test_prepared["households"]
)

X_test_cat = X_test[["ocean_proximity"]]

X_test_prepared = X_test_prepared.join(
    pd.get_dummies(X_test_cat, drop_first=True)
)
ingestion_logger.info("Data Ingestion - Validation Data - Prepared")

# user argument for output data path
parser = argparse.ArgumentParser()
parser.add_argument("--path", help="Output data path for processed data")
args = parser.parse_args()

if args.path:
    DATA_PATH = PARENT_PATH + "/" + args.path + "/"
else:
    DATA_PATH = PARENT_PATH + "/data/processed/"

# final dataset size
train_size = len(housing_prepared)
test_size = len(X_test_prepared)

# writing files
housing_prepared.to_csv(DATA_PATH + "housing_prepared.csv", index=False)
housing_labels.to_csv(DATA_PATH + "housing_labels.csv", index=False)
X_test_prepared.to_csv(DATA_PATH + "X_test_prepared.csv", index=False)
y_test.to_csv(DATA_PATH + "y_test.csv", index=False)
pickle.dump(imputer, open(PARENT_PATH + "/artifacts/imputer.pkl", "wb"))
ingestion_logger.info("Data Ingestion - Completed")
