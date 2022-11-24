""""""

#%% 
# Python imports

# Third party imports
from scipy.optimize import minimize
import numpy as np

# Local imports
from chiller_optimization.regression import load_model_pickled
from chiller_optimization.data_load import load_training_data_csv, load_dummy_data
from chiller_optimization.pipeline import linear_regression_pipeline

# Definitions

#%%

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
# x0 is the initial guess
x0 = np.array([0.1,1,1.2,0.8,2])
res = minimize(rosen, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True,})
print(res.x)

# %% Import linear model representing a single chiller
CUSTOMER_ID = 'testing'
MODEL_DIRECTORY = '../data/'
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

features, target, feature_names = load_dummy_data()
linear_regression_pipeline.fit(features)
x_transformed = linear_regression_pipeline.transform(x_raw)
power_input = linear_model.predict(x_transformed)
print('power input required: ', power_input)

# %%
