""""""

#%% 
# Python imports
from copy import deepcopy
import os
from typing import List, Callable, Tuple, List

# Third party imports
from scipy.optimize import (minimize, LinearConstraint, 
    NonlinearConstraint, Bounds)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from numpy.typing import ArrayLike

# Local imports
from chiller_optimization.regression import load_model_pickled, save_model_pickled, delete_model_pickled
from chiller_optimization.data_load import (
    load_training_data_csv, load_dummy_data, 
    REQUIRED_FILE_HEADERS, LINEAR_INPUT_SPECIFICATION)
from chiller_optimization.pipeline import linear_regression_pipeline
from chiller_optimization import parser

# Definitions
DEGREE: int = int(parser['pipeline']['degree_of_polynomial_features'])
ALPHA: float = float(parser['hyperparameters']['ridge_alpha'])
DATA_FILEPATH = '../data/generated_dummy_data2022-9-11.csv' # relative to parent package
TARGET_HEADER_NAME = 'power_input [kW]'
CUSTOMER_ID = 'testing'
MODEL_DIRECTORY = '../data/'
MIN_CAPACITY_OUTPUT = 9.84 # kW
MIN_CONDENSER_WATER_TEMPERATURE = 20.55 # DEG C
MIN_CONDENSER_WATER_FLOW_RATE = 0.9 # Percentage, 0-1 of rated design
MIN_EVAPORATOR_RETURN_WATER_TEMPERATURE = 7.8 # DEG C
MIN_EVAPORATOR_SUPPLY_WATER_TEMPERATURE = 4.44 # DEG C
MIN_EVPAORATOR_WATER_FLOW_RATE = 0.2 # Percentage, 0-1 of rated design
MAX_CAPACITY_OUTPUT = 49.2 # kW
MAX_CONDENSER_WATER_TEMPERATURE = 35 # DEG C
MAX_CONDENSER_WATER_FLOW_RATE = 1.1 # Percentage, 0-1 of rated design
MAX_EVAPORATOR_RETURN_WATER_TEMPERATURE = 20 # DEG C
MAX_EVAPORATOR_SUPPLY_WATER_TEMPERATURE = 12.78 # DEG C
MAX_EVPAORATOR_WATER_FLOW_RATE = 1.2 # Percentage, 0-1 of rated design

# %%

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
# x0 is the initial guess
x0 = np.array([0.1,1.3,1.2,0.8,2.6,5.7,6.1,7.2,8.3])

# Print results
np.set_printoptions(precision=2)
print(f"Example output of rosen function: {rosen(x0)}")
res = minimize(rosen, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True, 'maxiter':1e3})
print(f"Rosen function minimization with initial guess:\n{x0}")
print(f"Minimized result:\n{res.x}")

# Minimization with trust-constr method
x0 = np.array([0.1,1.3,1.2,0.8,2.6,5.7,6.1,7.2,8.3])
bounds = Bounds([0,1.5,-0.4,-0.3,0.5,0.4,0.3,-0.1,0.9], [1.0,2.0,0.8,0.9,0.7,0.6,0.5,0.4,0.95])
res1 = minimize(rosen, x0, method='trust-constr', bounds=bounds)
print("Rosen function minimization with initial guess:\n", x0)
print("Trust constrained method with bounds:")
for _lower, _upper in zip(bounds.lb, bounds.ub):
    print(_lower, " < x < ", _upper)
print("Minimized result: ", res1.x)
print("Bounds check")
for _lower, _upper, _minvalue in zip(bounds.lb, bounds.ub, res1.x):
    print(_lower, f" <= {_minvalue:0.2} <= ", _upper)

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

def minimize_wrapper(x0: Callable, *args) -> float:
    """Call the minimization function with a transformd input to the minimization function.
    Scipy minimize function expects a function callable with a single dimension input
    of size (n,). This function inputs an ArrayLike of shape (n,) and transforms it to shape
    (1,n) which is required for this specific linear model
    inputs
    -------
    x0: input to minimization function. Array of real elements of size (n,), 
        where n is the number of independent variables.
    *args: not used"""
    # Input dimenion required to be of shape (n,)
    # For this example, expecting (84,) becuase there are 84 features
    if x0.ndim != 1:
        msg="Expected input of 1 dimension. Got {} dimensions"
        raise ValueError(msg.format(result.ndim))

    x0_transformed = x0.reshape(1,-1)
    result: ArrayLike = linear_model.predict(x0_transformed) # Expected return vlaue is a 1 dimensional array
    
    # Prediction should be one dimensional with single prediction
    if result.shape != (1,):
        msg="Expected prediction function to return 1 dimensional array with shape (1,). Got {}"
        raise ValueError(msg.format(result.shape))

    return result[0]



function = minimize_wrapper# function to minimize
x0 = x_transformed # Initial guess
method = 'trust-constr'
bounds_description = { # Maximum and minimum according to specifications
    'capacity_output':[MIN_CAPACITY_OUTPUT, MAX_CAPACITY_OUTPUT],
    'condenser_water_temperature':[20.5, 35], # Exmple with multivariate bounds constraint
    'condenser_water_flow_rate': [0.9,1.1],
    'evaporator_water_return_temperature': [7.8, 20], 
    'evaporator_water_supply_temperature': [6.9, 6.9], # Fixed bounds [4.5, 12.7] 
    'evaporator_water_flow_rate': [0.9, 1.1],
}
minimum_bounds = np.array([x[0] for x in bounds_description.values()]).reshape(1,-1)
maximum_bounds = np.array([x[1] for x in bounds_description.values()]).reshape(1,-1)
minimum_bounds_transformed: ArrayLike = linear_regression_pipeline.transform(minimum_bounds).reshape(-1) # shape (84)
maximum_bounds_transformed: ArrayLike = linear_regression_pipeline.transform(maximum_bounds).reshape(-1) # shape (84)
bounds_transformed: List[List[float]] = []
# for minimum, maximum in zip(minimum_bounds_transformed[0], maximum_bounds_transformed[0]):
#     bounds_transformed.append([minimum, maximum])
bounds = Bounds(minimum_bounds_transformed, maximum_bounds_transformed)
results = minimize(function, x0, method=method, bounds=bounds)
print("Linear regression function minimization with initial guess:\n", x0)
print("Trust constrained method with bounds:")
for i in range(int(bounds.lb.shape[0] / 10)): # Print every 10th bounds
    print(f"{bounds.lb[i]:0.2}", " < x < ", f"{bounds.ub[i]:0.2}")
print("Minimized result: ", res1.x)
print("Bounds check")
for i in range(int(bounds.lb.shape[0] / 10)): # Print every 10th bounds
    print(f"{bounds.lb[i]:0.2}", f" <= {results.x[i]:0.2} <= ", f"{bounds.ub[i]:0.2}")

#%%

# Visualization and reverse
poly_features = linear_regression_pipeline.steps[0][1] # polynomial features
print("Features after scaling:")
print('capacity_output', f"{x0[0][1]:0.2}")
print('condenser_water_temperature', f"{x0[0][2]:0.2}")
print('condenser_water_flow_rate', f"{x0[0][3]:0.2}")
print('evaporator_water_return_temperature', f"{x0[0][4]:0.2}")
print('evaporator_water_supply_temperature', f"{x0[0][5]:0.2}")
print('evaporator_water_flow_rate', f"{x0[0][6]:0.2}")

standard_scaler = linear_regression_pipeline.steps[1][1] # Standard scaler
print('\nFeatures before scaling:')
mean_ = standard_scaler.mean_
stddev_ = np.sqrt(standard_scaler.var_)
print('capacity_output', f"{mean_[1] + x0[0][1] * stddev_[1]}")
print('condenser_water_temperature', f"{mean_[2] + x0[0][2] * stddev_[2]}")
print('condenser_water_flow_rate', f"{mean_[3] + x0[0][3] * stddev_[3]}")
print('evaporator_water_return_temperature', f"{mean_[4] + x0[0][4] * stddev_[4]}")
print('evaporator_water_supply_temperature', f"{mean_[5] + x0[0][5] * stddev_[5]}")
print('evaporator_water_flow_rate', f"{mean_[6] + x0[0][6] * stddev_[6]}")

print("\nminimized input features of linear regression:")
print('capacity_output', f"{mean_[1] + results.x[1] * stddev_[1]:0.3}")
print('condenser_water_temperature', f"{mean_[2] + results.x[2] * stddev_[2]:0.3}")
print('condenser_water_flow_rate', f"{mean_[3] + results.x[3] * stddev_[3]:0.3}")
print('evaporator_water_return_temperature', f"{mean_[4] + results.x[4] * stddev_[4]:0.3}")
print('evaporator_water_supply_temperature', f"{mean_[5] + results.x[5] * stddev_[5]:0.3}")
print('evaporator_water_flow_rate', f"{mean_[6] + results.x[6] * stddev_[6]:0.3}")

print("\nAllowable bounds of input to linear regression:")
names = ['capacity_output', 'condenser_water_temperature', 'condenser_water_flow_rate',
    'evaporator_water_return_temperature', 'evaporator_water_supply_temperature',
    'evaporator_water_flow_rate']
for i in range(1,7):
    print(names[i-1], f"{mean_[i] + bounds.lb[i] * stddev_[i]}", f" <= {mean_[i] + results.x[i] * stddev_[i]:0.3} <= ", f"{mean_[i] + bounds.ub[i] * stddev_[i]}")

# %% Zero finding with minimization of total power consumption of multiple devices

def minimize_wrapper_multiple_equipment(x0: Callable, *args) -> float:
    """Call the minimization function with a transformd input to the minimization function.
    Scipy minimize function expects a function callable with a single dimension input
    of size (n,). This function inputs an ArrayLike of shape (n,) and transforms it to shape
    (1,n) which is required for this specific linear model

    New constraints must be placed on models so total load is distributed between operable
    equipment. Sum of load distribution percentage must be == 1.

    inputs
    -------
    x0: input to minimization function. Array of real elements of size (n,), 
        where n is the number of independent variables.
    *args: [number of operable devices, transformation pipeline, n_features_each]"""
    # Input dimenion required to be of shape (6 * n_operable_devices)
    # For this example, expecting (6 * 2) becuase there are 6 features before transformation
    if x0.ndim != 1:
        msg="Expected input of 1 dimension. Got {} dimensions"
        raise ValueError(msg.format(result.ndim))

    if len(args) != 3:
        msg=("Expected number of operable devices, tranformation pipeline, " +
            "and n_features_each to be passed to function wrapper. Got {}")
        raise ValueError(msg.format(len(args)))

    if args[0] <= 1:
        msg="Expected multiple equipment for this callable. Got {}"
        raise ValueError(msg.format(args[0]))

    if not isinstance(args[1], Pipeline):
        msg="Expected transformation pipeline to be passed as argument. Got {}"
        raise ValueError(msg.format(args[1]))

    # Tranform input vector into format acceptable for regression
    x0 = x0.reshape(args[0],args[2]) # Transform into matrix of shape (number of operable devices, number of features per device)
    x0_transformed = args[1].transform(x0) # Apply transformation along first axis
    # Expected return vlaue is a 1 dimensional array with shape (number of operable devices)
    prediction: ArrayLike = linear_model.predict(x0_transformed) 
    
    # Prediction should be one dimensional with single prediction
    if prediction.shape != (args[0],):
        msg="Expected prediction function to return 1 dimensional array with shape (1,). Got {}"
        raise ValueError(msg.format(prediction.shape))

    return np.sum(prediction, axis=0)

def determine_bounds_single_equipment(
    minimum_capacity_output:float,
    maximum_capacity_output:float,
    condenser_water_temperature:float,
    condenser_water_flow_rate:float,
    evaporator_water_return_temperature:float,
    evaporator_water_supply_temperature:float,
    evaporator_water_flow_rate:float) -> Tuple[List[float], List[float]]:
    """Output bounds which represent the allowable operating ranges of a single piece
    of equipment
    Expect 6 input features. Of these input features, (5) remain constant, while capacity
    output can vary between minimum and maximum rated capacity
    Input features:
    'capacity_output': (float) between minimum and maximum rated for equipment
    'condenser_water_temperature': (float) between (20.5, 35)
    'condenser_water_flow_rate': (float) between (0.9, 1.1)
    'evaporator_water_return_temperature': (float) between (7.8, 20)
    'evaporator_water_supply_temperature': (float) between (4.44, 12.78)
    'evaporator_water_flow_rate': (float) between (0.2, 1.2)
    """

    bounds_description = { # Maximum and minimum according to specifications
    'capacity_output':[minimum_capacity_output, maximum_capacity_output],
    'condenser_water_temperature':[condenser_water_temperature, condenser_water_temperature], # static
    'condenser_water_flow_rate': [condenser_water_flow_rate, condenser_water_flow_rate], # static
    'evaporator_water_return_temperature': [evaporator_water_return_temperature,evaporator_water_return_temperature], # static
    'evaporator_water_supply_temperature': [evaporator_water_supply_temperature,evaporator_water_supply_temperature], # static
    'evaporator_water_flow_rate': [evaporator_water_flow_rate, evaporator_water_flow_rate], # static
    }
    minimum_bounds = [x[0] for x in bounds_description.values()]
    maximum_bounds = [x[1] for x in bounds_description.values()]

    return (minimum_bounds, maximum_bounds)

# Number of operable equipmnt (chillers) and hyperparameters
N_EQUIPMENT = 2
TARGET_CAPACITY_OUTPUT = 1.63 * MAX_CAPACITY_OUTPUT
METHOD = 'trust-constr'
N_FEATURES_PER_EQUIPMENT:int = len(LINEAR_INPUT_SPECIFICATION)
x0 = np.array(
    # Format [capacity_output, condenser_water_temperature, condenser_water_flow_rate, evaporator_water_return_temperature,
    # evaporator_water_supply_temperature, evaporator_water_flow_rate]
    [20, 20.5, 0.95, 12.43, 7.65, 0.78, # Features for first equipment
    20, 20.5, 0.95, 12.43, 7.65, 0.78] # Features for second equipment
    )

# Define minimization function and bounds
function = minimize_wrapper_multiple_equipment # function to minimize

# Bounds definition and determination
equipment1_min_bounds, equipment1_max_bounds = determine_bounds_single_equipment(
    MIN_CAPACITY_OUTPUT,
    MAX_CAPACITY_OUTPUT,
    condenser_water_temperature=20.5,
    condenser_water_flow_rate=0.95,
    evaporator_water_return_temperature=12.43,
    evaporator_water_supply_temperature=7.65,
    evaporator_water_flow_rate=0.78,
)
equipment2_min_bounds, equipment2_max_bounds = determine_bounds_single_equipment(
    MIN_CAPACITY_OUTPUT+2, # Slightly differnet piece of equipment with different operation ranges
    MAX_CAPACITY_OUTPUT+2,
    condenser_water_temperature=20.5,
    condenser_water_flow_rate=0.95,
    evaporator_water_return_temperature=12.43,
    evaporator_water_supply_temperature=7.65,
    evaporator_water_flow_rate=0.78,
)
minimum_bounds_transformed = list((*equipment1_min_bounds, *equipment2_min_bounds))
maximum_bounds_transformed = list((*equipment1_max_bounds, *equipment2_max_bounds))
bounds = Bounds(minimum_bounds_transformed, maximum_bounds_transformed) # size (12,)

# Total capacity output must match target
"""
The bounds definition and linear constraint definitions change based on the number of
Operable pieces of equipment.
The bounds definition and linear constraint must have size (6 * N_EQUIPMENT)
Because each equipment has 6 prediction features, and each equipment must have
bounds definitions for that equipment. For example, the capacity output
of two different equipment may not match
"""
INPUT_SPECIFICATION_SIZE: int = N_FEATURES_PER_EQUIPMENT * N_EQUIPMENT # 12
constraint_matrix = np.zeros(shape=(INPUT_SPECIFICATION_SIZE,))
for i in range(0, constraint_matrix.size):
    if i % N_FEATURES_PER_EQUIPMENT == 0:
        # Linear constraint only applies to capacity output target
        constraint_matrix[i] = 1 
lower_bound = np.array([TARGET_CAPACITY_OUTPUT])
upper_bound = np.array([TARGET_CAPACITY_OUTPUT])
linear_constraint = LinearConstraint(constraint_matrix, lower_bound, upper_bound)

# Run minimization
results = minimize(function, x0, method=METHOD, bounds=bounds,
    args=(N_EQUIPMENT, linear_regression_pipeline, N_FEATURES_PER_EQUIPMENT),
    constraints=(linear_constraint),
    )
print("Linear regression function minimization with initial guess:\n", x0)
print("Trust constrained method with bounds:")
for i in range(bounds.lb.shape[0]): # Print every bounds
    print(f"{bounds.lb[i]:0.2f}", " < x < ", f"{bounds.ub[i]:0.2f}", end=" ")
    print(LINEAR_INPUT_SPECIFICATION[i % N_FEATURES_PER_EQUIPMENT][0])
print("Minimized result: ", results.x)
print("Bounds check")
for i in range(bounds.lb.shape[0]):
    print(f"{bounds.lb[i]:0.2f}", f" <= {results.x[i]:0.2f} <= ", f"{bounds.ub[i]:0.2f}")
print("\nConstraint check")
print("Total cooling requirement: ", TARGET_CAPACITY_OUTPUT)
print(f"{linear_constraint.lb[0]:0.2f}", f" <= {results.x[0] + results.x[6]:0.2f} <= ", f"{linear_constraint.ub[0]:0.2f}")

# %%

# delete the saved model once we are done with it
os.remove(model_filepath)
