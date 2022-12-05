import os
import tarfile
import pandas as pd
from six.moves import urllib

# path to files in package
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
PARENT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
HOUSING_PATH = os.path.join(PARENT_PATH + "/data/raw", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


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


def test_data_download():
    fetch_housing_data()
    assert os.path.exists(HOUSING_PATH)
    assert os.path.exists(HOUSING_PATH + "/housing.csv")


def test_data_load():
    test_data = load_housing_data()
    assert isinstance(test_data, pd.DataFrame)
