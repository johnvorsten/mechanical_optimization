""""""

#%% 
# Python imports
from copy import deepcopy
import os
from typing import List

# Third party imports
from scipy.optimize import (minimize, LinearConstraint, 
    NonlinearConstraint, Bounds)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

# Local imports
from chiller_optimization.regression import load_model_pickled, save_model_pickled, delete_model_pickled
from chiller_optimization.data_load import load_training_data_csv, load_dummy_data, REQUIRED_FILE_HEADERS
from chiller_optimization.pipeline import linear_regression_pipeline
from chiller_optimization import parser

# Definitions
DEGREE: int = int(parser['pipeline']['degree_of_polynomial_features'])
ALPHA: float = float(parser['hyperparameters']['ridge_alpha'])
DATA_FILEPATH = '../data/generated_dummy_data2022-9-11.csv' # relative to parent package
TARGET_HEADER_NAME = 'power_input [kW]'
CUSTOMER_ID = 'testing'
MODEL_DIRECTORY = '../data/'
MAX_CAPACITY_OUTPUT = 49.2
MIN_CAPACITY_OUTPUT = 9.84


#%%

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
# x0 is the initial guess
x0 = np.array([0.1,1.3,1.2,0.8,2.6,5.7,6.1,7.2,8.3])
print(f"Example output of rosen function: {rosen(x0)}")
res = minimize(rosen, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True, 'maxiter':1e3})
print(f"Rosen function minimization with initial guess:\n{x0}")
print(f"Minimized result:\n{res.x}")

# %% 
## Create a linear model representing a single chiller
# Load data
features, target, feature_headers = load_dummy_data()

# Data pipeline
X = linear_regression_pipeline.fit_transform(features)

# Split data into training and testing slices
# Shuffle data loaded
x_train, x_test, y_train, y_test= train_test_split(X, target, test_size=0.15, shuffle=True)

# Define a model
linear_model = LinearRegression(fit_intercept=True)
linear_model.fit(x_train, y_train)

# Train a model & hyperparameter tuning
cv_results_linear = cross_validate(linear_model, x_train, y_train, cv=3, scoring=['explained_variance','max_error','neg_mean_squared_error'])

# Save model
model_filepath = save_model_pickled(CUSTOMER_ID, MODEL_DIRECTORY, linear_model)
del linear_model # Sanity

# Import linear model representing a single chiller
linear_model = load_model_pickled(CUSTOMER_ID, MODEL_DIRECTORY)

# Make a prediction with some static data
capacity_output = 10.21 # kW
condenser_water_temperature = 29 # DEG C
condenser_water_flow_rate = 1.0 # [percent]
evaporator_water_return_temperature = 14.5 # [DEG C]
evaporator_water_supply_temperature = 9.0 # [DEG C]
evaporator_water_flow_rate = 1.0 # [percent]
x_raw = np.array([[
    capacity_output, condenser_water_temperature, condenser_water_flow_rate,
    evaporator_water_return_temperature, evaporator_water_supply_temperature,
    evaporator_water_flow_rate,
]])

# Re-load data for tutorial
features, target, feature_names = load_dummy_data()
linear_regression_pipeline.fit(features)
x_transformed = linear_regression_pipeline.transform(x_raw)
power_input = linear_model.predict(x_transformed)
print(f'power input required: {power_input} [kW]')
print('Expected approximately 6 [kW]')

# %% Zero finding
fun = linear_model.predict # function to minimize
x0 = x_transformed # Initial guess
method = 'trust-constr'
bounds_description = { # Maximum and minimum according to specifications
    'capacity_output':[MIN_CAPACITY_OUTPUT, MAX_CAPACITY_OUTPUT],
    'condenser_water_temperature':[20.5, 35], 
    'condenser_water_flow_rate': [0.9,1.1],
    'evaporator_water_return_temperature': [7.8, 20], 
    'evaporator_water_supply_temperature': [4.5, 12.7],
    'evaporator_water_flow_rate': [0.9, 1.1],
}
minimum_bounds = np.array([x[0] for x in bounds_description.values()]).reshape(1,-1)
maximum_bounds = np.array([x[1] for x in bounds_description.values()]).reshape(1,-1)
minimum_bounds_transformed = linear_regression_pipeline.transform(minimum_bounds) 
maximum_bounds_transformed = linear_regression_pipeline.transform(maximum_bounds)
bounds_transformed: List[List[float]] = []
for minimum, maximum in zip(minimum_bounds_transformed, maximum_bounds_transformed):
    bounds_transformed.append([minimum, maximum])
bounds = Bounds(*bounds_transformed)
results = minimize(fun, x0, method=method, bounds=bounds_transformed)

# %%

# delete the saved model once we are done with it
os.remove(model_filepath)
