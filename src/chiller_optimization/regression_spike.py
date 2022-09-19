"""Given a set of data, create a regression model which estimates the target
variable (power input) given a set of independent variables"""
#%%

# Python imports
from configparser import ConfigParser
from typing import Union, Any, List, Dict
from datetime import datetime, timezone
import time
import pickle
import os
import csv
from copy import deepcopy

# Third party imports
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_validate
from matplotlib import pyplot as plt
from sklearn.metrics import (explained_variance_score, mean_absolute_error, mean_squared_error, max_error)

# Local imports
from pipeline import linear_regression_pipeline
from data_load import load_training_data_csv, REQUIRED_FILE_HEADERS

# Declaration
parser = ConfigParser()
parser.read('./config.ini')
DEGREE: int = int(parser['pipeline']['degree_of_polynomial_features'])
ALPHA: float = float(parser['hyperparameters']['ridge_alpha'])
DATA_FILEPATH = '../../data/generated_dummy_data2022-9-11.csv'
TARGET_HEADER_NAME = 'power_input [kW]'


#%%

# Load data
data = load_training_data_csv(DATA_FILEPATH, headers=REQUIRED_FILE_HEADERS)
features_headers: List[str] = deepcopy(REQUIRED_FILE_HEADERS)
features_headers.remove('power_input [kW]')
features_index: List[int] = list(range(len(REQUIRED_FILE_HEADERS)))
features_index.remove(REQUIRED_FILE_HEADERS.index('power_input [kW]'))
features = data[:,features_index]
target_index = REQUIRED_FILE_HEADERS.index('power_input [kW]')
target = data[:, target_index]

# Data pipeline
X = linear_regression_pipeline.fit_transform(features)

# Split data into training and testing slices
# Shuffle data loaded
x_train, x_test, y_train, y_test= train_test_split(X, target, test_size=0.15, shuffle=True)

# Define a model
linear_model = LinearRegression(fit_intercept=True)
ridge_model = Ridge(fit_intercept=True, alpha=ALPHA)

# Train a model & hyperparameter tuning
cv_results_linear = cross_validate(linear_model, x_train, y_train, cv=3, scoring=['explained_variance','max_error','neg_mean_squared_error'])
cv_results_ridge = cross_validate(ridge_model, x_train, y_train, cv=3, scoring=['explained_variance','max_error','neg_mean_squared_error'])

# Model performance metrics
print("Linear model cross validation results:")
for key, value in cv_results_linear.items():
    print(key, value)
print("Ridge model cross validation results:")
for key, value in cv_results_ridge.items():
    print(key, value)

# Performance of model on test (validation) slice
linear_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
y_pred_linear_test = linear_model.predict(x_test)
y_pred_ridge_test = ridge_model.predict(x_test)
results_linear: Dict[str, int] = {}
results_ridge: Dict[str, int] = {}
results_linear['explained_variance'] = explained_variance_score(y_test, y_pred_linear_test)
results_linear['mean_absolute'] = mean_absolute_error(y_test, y_pred_linear_test)
results_linear['mean_squared'] = mean_squared_error(y_test, y_pred_linear_test)
results_linear['max_error'] = max_error(y_test, y_pred_linear_test)
results_ridge['explained_variance'] = explained_variance_score(y_test, y_pred_ridge_test)
results_ridge['mean_absolute'] = mean_absolute_error(y_test, y_pred_ridge_test)
results_ridge['mean_squared'] = mean_squared_error(y_test, y_pred_ridge_test)
results_ridge['max_error'] = max_error(y_test, y_pred_ridge_test)

print("Linear model test validation results:")
for key, value in results_linear.items():
    print(key, value)
print("Ridge model test validation results:")
for key, value in results_ridge.items():
    print(key, value)

#%%
# Visualiation of performance metrics
# Minimum capacity output (9.84kW) condenser water temperature ranges 20-35 DEG C
DATA_START = 0 # 0-indexed row from .csv file
DATA_END = 50 # closed end
condenser_water_temperature: np.ndarray = features[DATA_START:DATA_END, features_headers.index('condenser_water_temperature [DEG C]')]
y_pred = linear_model.predict(X[DATA_START:DATA_END,:])
y_true = target[DATA_START:DATA_END]
fix, ax = plt.subplots()
ax.scatter(condenser_water_temperature, y_true, label='target data')
ax.scatter(condenser_water_temperature, y_pred, label='predicted')
ax.set_xlabel('Condenser water temperature [DEG C]')
ax.set_ylabel('Power input [kW]')
ax.set_title('Predicted power input at minimum cooling output\nvarying condenser water temperature')
ax.legend()

# Minimum capacity output (9.84kW) condenser water flow rate ranges 90%-110%
DATA_START = 2501 # 0-indexed row from .csv file
DATA_END = 2550 # closed end
condenser_water_flow_rate: np.ndarray = features[DATA_START:DATA_END, features_headers.index('condenser_water_flow_rate [percent]')]
y_pred = linear_model.predict(X[DATA_START:DATA_END,:])
y_true = target[DATA_START:DATA_END]
fix, ax = plt.subplots()
ax.scatter(condenser_water_flow_rate, y_true, label='target data')
ax.scatter(condenser_water_flow_rate, y_pred, label='predicted')
ax.set_xlabel('Condenser water flow rate [%]')
ax.set_ylabel('Power input [kW]')
ax.set_title('Predicted power input at minimum cooling output\nvarying condenser water flow rate')
ax.legend()

# Some random data
rng = np.random.default_rng()
DATA_INDEX: np.ndarray = rng.integers(low=0, high=X.shape[0], size=50)
capacity_output: np.ndarray = features[DATA_INDEX, features_headers.index('capacity_output [kW]')]
y_pred = linear_model.predict(X[DATA_INDEX,:])
y_true = target[DATA_INDEX]
fix, ax = plt.subplots()
ax.scatter(capacity_output, y_true, label='target data')
ax.scatter(capacity_output, y_pred, label='predicted')
ax.set_xlabel('Condenser water flow rate [%]')
ax.set_ylabel('Power input [kW]')
ax.set_title('Predicted power input at minimum cooling output\nvarying condenser water flow rate')
ax.legend()

# %%
