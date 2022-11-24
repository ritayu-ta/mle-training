# Managing the end-to-end machine learning lifecycle with MLFlow

This Repository contains the resources for my tutorial **"Managing the end-to-end machine learning lifecycle with MLFlow"** at pyData/pyCon Berlin 2019.

# Basic setup

## Setup the environment
- clone this repository
- **with virtualenv (recommended)**
  - install virtualenv: `pip install virtualenv`
  - create a new environment: `virtualenv mlflow_tutorial`
  - activate the environment: `source mlflow_tutorial/bin/activate`
  - run `pip install -r requirements.txt`

## The notebook
- Get the `mlflow-example.ipynb`
- run `jupyter notebook`

## Command to setup mlflow server
- `mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host localhost --port 5000`
