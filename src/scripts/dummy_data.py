"""
Create B-splines which represent dummy equipment operating curves
"""
#%%
# Python imports
from typing import Tuple, List
import csv
from collections import namedtuple

# Third party imports
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator, interp1d
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# Local imports

# Declarations
rng = np.random.default_rng()
# Interpolations
data_points = np.array([ # condenser water temperature [DEG C], power input[kW], capacity output [kW]
    [35.01, 21.9, 49.20], # maximum capacity, COP 2.246
    [35.02, 21.87, 49.19], # COP 2.246
    [35.02, 21.85, 49.15], # COP ~2.246
    [35.02, 21.82, 49.11], # COP ~2.246
    [35.02, 6.84, 9.84], # Minimum capacity, COP 1.43
    [35.02, 6.90, 9.93], # COP 1.43 
    [34.99, 9.15, 32.40], # COP ~3.54
    [34.99, 9.25, 33.40], # COP ~3.61
    [34.99, 9.37, 34.40], # highest efficiency, COP 3.67
    [35.01, 9.39, 34.39], # COP ~3.67
    [35.01, 9.42, 34.40], # COP ~3.67
    [35.01, 9.45, 34.40], # COP ~3.67
    [35.01, 9.48, 34.41], # COP ~3.67
    [34.10, 7.46, 19.40], # other point, COP 2.8
    [35.02, 7.50, 19.45], # COP ~2.8
    [35.01, 7.50, 19.47], # COP ~2.8
    [34.12, 7.51, 19.49], # COP ~2.8
    ], dtype=np.float32)
independent_data = data_points[:,[0,2]]
target = data_points[:,1]
interpolator = RBFInterpolator(independent_data, target, smoothing=1, degree=3, kernel='thin_plate_spline')

# File import
DUMMY_DATA_FILEPATH = '../../data/dummy_data.csv'
DUMMY_DATA = np.genfromtxt(DUMMY_DATA_FILEPATH, delimiter=',')
chiller_1_data = DUMMY_DATA[:, [1,2,5]]
chiller_2_data = DUMMY_DATA[:, [1,3,6]]
chiller_3_data = DUMMY_DATA[:, [1,4,7]]

# File saving
SAVED_DATA_FILEPATH = '../../data/generated_dummy_data.csv'

# Independnet variables definition
INDEPENDENT_VARIABLES: List[str] = [
    "Cooling load (power) required [kW]",
    "Condenser water temperature [DEG C]",
    "Condenser water flow rate (percent of rated for chiller) [percent]",
    "Number of operable chillers [integer]",
    "Evaporator return water temperature [DEG C]",
    "Evaporator supply water temperature (setpoint) [DEG C]",
    "Evaporator water flow rate (percent of rated for chiller) [percent]",
]
FILE_HEADERS: List[str] = [
    'capacity_output [kW]',
    'power_input [kW]',
    'condenser_water_temperature [DEG C]',
    'condenser_water_flow_rate [percent]',
    'evaporator_water_return_temperature [DEG C]',
    'evaporator_water_supply_temperature [DEG C]',
    'evaporator_water_flow_rate [percent]',
    ]

# Regressor and data constants
N_SAMPLES = 50
# Generate some dummy data for each of the independent variables
"""
Define a dummy distribution for each of the variables below, which will be
used to influence how much input power is required. Each independnet variable
will have a tuple of values representing its process value (how much it is measured to
be as a physical measurement like temperature) and how much it influences power
consumption.
Example - condenser water temperatures [(25, -0.01), (28, -0.01), (30, 0.00), (32, 0.01), (35, 0.02)]
This can be interpreted as at 25 DEG C, the condenser water temperature will reduce power input by 1%.
At 30 DEG C the condenser water temperautre will have no effect on power input. At 35 DEG C power input will
be increased by 2%

Independent variables:
* Cooling load (power) required [kW]
* Condenser water temperature [DEG C]
* Condenser water flow rate [percent of rated for chiller]
* Number of operable chillers [integer]
* Evaporator return water temperature
* Evaporator supply water temperature (setpoint)
* Evaporator water flow rate [percent of rated for chiller]
"""

CONDENSER_WATER_TEMPERATURE_SCALE = [
    (20.55, -0.045), (22.77, -0.03), (25, -0.02), (28, -0.01), (30, 0.00), (32, 0.01), (35, 0.02)] # (degrees C, percent)
CONDENSER_WATER_FLOW_RATE_SCALE = [
    (0.9, 0.005), (1.0, 0.0), (1.1, -0.01)] # (percent, percent)
EVAPORATOR_RETURN_WATER_TEMPERATURE_SCALE = [
    (7.78,0.10), (9.14,0.09), (10.50,0.065), (11.85,0.03), (13.21,0.01), (14.57,0.0), (15.93,-0.01), (17.28,-0.02), (18.64,-0.03), (20.0,-0.04)] # (degrees C, percent) 46 to 68 DEG F
EVAPORATOR_SUPPLY_WATER_TEMPERATURE_SCALE = [
    (4.44,0.05), (5.37,0.04), (6.29,0.03), (7.22,0.02), (8.15,0.00), (9.07,-0.01), (10.0,-0.03), (10.93,-0.04), (11.85,-0.05), (12.78,-0.06)] # (degrees C, percent) 40 to 55 deg F
EVPAORATOR_WATER_FLOW_RATE_SCALE = [
    (0.2,0.02), (0.3,0.02), (0.4,0.00), (0.5,0.0), (0.6,0.0), (0.7,0.0), (0.8,0.0), (0.9,0.0), (1.0,0.0), (1.1,0.0), (1.2,0.01)] # (percent, percent)

DEFAULT_CONDENSER_WATER_TEMPERATURE = 30 # Degrees celisius, 86 DEG F
DEFAULT_CONDENSER_WATER_FLOW_RATE = 1.0 # 100% of rated condenser water flow
DEFAULT_EVAPORATOR_RETURN_WATER_TEMPERATURE = 14.57 # Degrees Celsius, 58.5 DEG F
DEFAULT_EVAPORATOR_SUPPLY_WATER_TEMPERATURE = 8.5 # Degrees Celsius, 47.3 DEG F
DEFAULT_EVAPORATOR_WATER_FLOW_RATE = 1.0 # 100% rated evaporator flow
MINIMUM_COOLING_OUTPUT = 9.84
MIN_CONDENSER_WATER_TEMPERATURE = 20.55
MIN_CONDENSER_WATER_FLOW_RATE = 0.9
MIN_EVAPORATOR_RETURN_WATER_TEMPERATURE = 7.8
MIN_EVAPORATOR_SUPPLY_WATER_TEMPERATURE = 4.44
MIN_EVPAORATOR_WATER_FLOW_RATE = 0.2
MAXIMUM_COOLING_OUTPUT = 49.2
MAX_CONDENSER_WATER_TEMPERATURE = 35
MAX_CONDENSER_WATER_FLOW_RATE = 1.1
MAX_EVAPORATOR_RETURN_WATER_TEMPERATURE = 20
MAX_EVAPORATOR_SUPPLY_WATER_TEMPERATURE = 12.78
MAX_EVPAORATOR_WATER_FLOW_RATE = 1.2

#%%
"""
Interpolation (line must go through given points) of capacity output versus power input
"""
xs = np.linspace(start=MINIMUM_COOLING_OUTPUT, stop=MAXIMUM_COOLING_OUTPUT, num=50)
ys = interpolator(np.stack(
            (np.array([35]*50, dtype=np.float32), xs), axis=1)
            )
efficiency = xs / ys
fig, ax = plt.subplots()
ax_efficiency = ax.twinx()
ax.scatter(xs, ys)
ax_efficiency.plot(xs, efficiency, c='red', label='efficiency')
ax.axvline(x=MINIMUM_COOLING_OUTPUT, dashes=(5,5))
ax.axvline(x=MAXIMUM_COOLING_OUTPUT, dashes=(5,5))
ax.text(x=MINIMUM_COOLING_OUTPUT, y=15, s='minimum\ncapacity')
ax.text(x=MAXIMUM_COOLING_OUTPUT, y=15, s='maximum\ncapacity')
ax.set_xlabel('Capacity output (kW of cooling)')
ax.set_ylabel('Electric power input (kW)')
ax_efficiency.set_ylabel('Efficiency [kWA/kW]')
ax.set_title('Interpolation of cooling capacity output ')
ax_efficiency.legend(loc='lower center')
plt.show()


#%%

def random_noise(min:float, max:float) -> float:
    return (max - min) * rng.random() + min

def random_noise_standard_normal(min:float, max:float) -> float:
    return (max - min) * rng.standard_normal() + min

pipeline = Pipeline(
    [
    ('polynomial', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ]
)

# Scale the dependent data based on our weights to create dummy data
regression_data_points = np.array([ # power input[kW], capacity output [kW]
    [21.9, 49.20], # maximum capacity, COP 2.246
    [6.84, 9.84], # Minimum capacity, COP 1.43
    [9.37, 34.40], # highest efficiency, COP 3.67
    [7.50, 19.45], # COP ~2.8
    ], dtype=np.float32)
desired_capacity = np.array(regression_data_points[:,1]).reshape(-1,1) # Capacity output (desired output capacity)
dependent_data = regression_data_points[:,0] # Power input (given the desired output capacity, power input is predicted)
linear_regressor_power_input = LinearRegression(fit_intercept=True, n_jobs=2)
X = pipeline.fit_transform(desired_capacity)
y = dependent_data
linear_regressor_power_input.fit(X, y)

"""Linear regression of power input and cooling power output"""
x_scale = np.linspace(start=MINIMUM_COOLING_OUTPUT, stop=MAXIMUM_COOLING_OUTPUT, num=N_SAMPLES).reshape(-1,1) # Cooling power output [kW]
x_scale_2 = np.linspace(start=9, stop=60, num=N_SAMPLES).reshape(-1,1)
y_hat = linear_regressor_power_input.predict(pipeline.fit_transform(x_scale)) # Predicted power input (N_SAMPLES, 1) [kW]
efficiency = np.divide(x_scale.reshape(-1), y_hat)
fig, ax = plt.subplots()
ax_efficiency = ax.twinx()
ax.scatter(x_scale, y_hat)
ax.scatter(desired_capacity, y)
ax_efficiency.plot(x_scale, efficiency, c='red', label='efficiency')
ax.axvline(x=MINIMUM_COOLING_OUTPUT, dashes=(5,5))
ax.axvline(x=MAXIMUM_COOLING_OUTPUT, dashes=(5,5))
ax.text(x=MINIMUM_COOLING_OUTPUT, y=25, s='minimum\ncapacity')
ax.text(x=MAXIMUM_COOLING_OUTPUT, y=25, s='maximum\ncapacity')
ax.set_xlabel('Capacity output (kW of cooling)')
ax.set_ylabel('Electric power input (kW)')
ax_efficiency.set_ylabel('Efficiency [kWA/kW]')
ax.set_title('Linear regression of cooling capacity output ')
ax_efficiency.legend(loc='lower center')
plt.show()

#%%

# For each independent variable, scale the dependent data and save the generated data
# Condenser water temperature linear regression
independent_data = np.array([val for (val, _) in CONDENSER_WATER_TEMPERATURE_SCALE]).reshape(-1,1) # condenser water temp
y = np.array([scale for (_, scale) in CONDENSER_WATER_TEMPERATURE_SCALE]).reshape(-1,1) # Scaling factor
condenser_water_temperature_regressor = LinearRegression(fit_intercept=True, n_jobs=2)
X = pipeline.fit_transform(independent_data)
condenser_water_temperature_regressor.fit(X, y) # output scale factor based on condenser water temperature

# Condenser water flow rate linear regression
independent_data = np.array([val for (val, _) in CONDENSER_WATER_FLOW_RATE_SCALE]).reshape(-1,1) # condenser water flow rate
y = np.array([scale for (_, scale) in CONDENSER_WATER_FLOW_RATE_SCALE]).reshape(-1,1) # Scaling factor
condenser_water_flow_regressor = LinearRegression(fit_intercept=True, n_jobs=2)
X = pipeline.fit_transform(independent_data)
condenser_water_flow_regressor.fit(X, y) # output scale factor

# evaporator water return temperature linear regression
independent_data = np.array([val for (val, _) in EVAPORATOR_RETURN_WATER_TEMPERATURE_SCALE]).reshape(-1,1) # evaporator water return temperature
y = np.array([scale for (_, scale) in EVAPORATOR_RETURN_WATER_TEMPERATURE_SCALE]).reshape(-1,1) # Scaling factor
evaporator_water_return_temperature_regressor = LinearRegression(fit_intercept=True, n_jobs=2)
X = pipeline.fit_transform(independent_data)
evaporator_water_return_temperature_regressor.fit(X, y) # output scale factor

# evaporator water supply temperature linear regression
independent_data = np.array([val for (val, _) in EVAPORATOR_SUPPLY_WATER_TEMPERATURE_SCALE]).reshape(-1,1) # evaporator water supply temperature
y = np.array([scale for (_, scale) in EVAPORATOR_SUPPLY_WATER_TEMPERATURE_SCALE]).reshape(-1,1) # Scaling factor
evaporator_water_supply_temperature_regressor = LinearRegression(fit_intercept=True, n_jobs=2)
X = pipeline.fit_transform(independent_data)
evaporator_water_supply_temperature_regressor.fit(X, y) # output scale factor

# evaporator water flow rate linear regression
independent_data = np.array([val for (val, _) in EVPAORATOR_WATER_FLOW_RATE_SCALE]).reshape(-1,1) # evaporator water flow rate
y = np.array([scale for (_, scale) in EVPAORATOR_WATER_FLOW_RATE_SCALE]).reshape(-1,1) # Scaling factor
evaporator_water_flow_regressor = LinearRegression(fit_intercept=True, n_jobs=2)
X = pipeline.fit_transform(independent_data)
evaporator_water_flow_regressor.fit(X, y) # output scale factor

#%%

# Baseline data for prediction
capacity_output = np.linspace(start=MINIMUM_COOLING_OUTPUT, stop=MAXIMUM_COOLING_OUTPUT, num=N_SAMPLES).reshape(-1,1) # (50,1) power output [kW]
baseline_power_input = linear_regressor_power_input.predict(pipeline.fit_transform(capacity_output)).reshape(-1,1) # power input (N_SAMPLES,) [kW]
baseline_power_input_grid = np.broadcast_to(baseline_power_input, (N_SAMPLES, N_SAMPLES, len(INDEPENDENT_VARIABLES)-1))
scaled_power_input = np.copy(baseline_power_input_grid) # scaled power input required (N_SAPMLES,N_SAMPLES, 6)

"""Linear regression of power input and cooling power output"""
fig, ax = plt.subplots()
ax.scatter(capacity_output, baseline_power_input)
ax.legend(['Linear regression baseline power input'])
ax.set_title("Linear regression baseline power input v. capacity output")
ax.set_xlabel("Capacity output [kW]")
ax.set_ylabel("Power input [kW]")
ax.axvline(x=MINIMUM_COOLING_OUTPUT, dashes=(5,5))
ax.axvline(x=MAXIMUM_COOLING_OUTPUT, dashes=(5,5))
ax.text(x=MINIMUM_COOLING_OUTPUT, y=15, s='minimum\ncapacity')
ax.text(x=MAXIMUM_COOLING_OUTPUT, y=15, s='maximum\ncapacity')

# Scaled power input multiplier for condenser water temperature
xnew_condenser_water_temperature = np.linspace(CONDENSER_WATER_TEMPERATURE_SCALE[0][0], CONDENSER_WATER_TEMPERATURE_SCALE[-1][0], num=N_SAMPLES).reshape(-1,1) # Condenser water temperature
scale_condenser_water_temperature = condenser_water_temperature_regressor.predict(pipeline.fit_transform(xnew_condenser_water_temperature)).reshape(-1) # Power scale percent (N_SAMPLES,)

"""
Example of how input power scales with an independent variable changing
"""
fig, ax = plt.subplots()
ax.plot(xnew_condenser_water_temperature, scale_condenser_water_temperature, '--')
ax.legend(['Result of power input'], loc='best')
ax.set_title("Example of condenser water temperature\non power input required")
ax.set_xlabel("Condenser water temperature [DEG C]")
ax.set_ylabel("Power input scale\nscaled_power_input = power_input * (1+y) [%]")
plt.show()

# Scaled power input multiplier for condenser water flow rate
xnew_condenser_water_flow_rate = np.linspace(CONDENSER_WATER_FLOW_RATE_SCALE[0][0], CONDENSER_WATER_FLOW_RATE_SCALE[-1][0], num=N_SAMPLES).reshape(-1,1)
scale_condenser_water_flow = condenser_water_flow_regressor.predict(pipeline.fit_transform(xnew_condenser_water_flow_rate)).reshape(-1)

# Scaled power input multiplier for evaporator water return temperature
xnew_evaporator_water_return_temperature = np.linspace(EVAPORATOR_RETURN_WATER_TEMPERATURE_SCALE[0][0], EVAPORATOR_RETURN_WATER_TEMPERATURE_SCALE[-1][0], num=N_SAMPLES).reshape(-1,1)
scale_evaporator_water_return_temperature = evaporator_water_return_temperature_regressor.predict(pipeline.fit_transform(xnew_evaporator_water_return_temperature)).reshape(-1)

# Scaled power input multiplier for evaporator water supply temperature
xnew_evaporator_water_supply_temperature = np.linspace(EVAPORATOR_SUPPLY_WATER_TEMPERATURE_SCALE[0][0], EVAPORATOR_SUPPLY_WATER_TEMPERATURE_SCALE[-1][0], num=N_SAMPLES).reshape(-1,1)
scale_evaporator_water_supply_temperature = evaporator_water_supply_temperature_regressor.predict(pipeline.fit_transform(xnew_evaporator_water_supply_temperature)).reshape(-1)

# Scaled power input multiplier for evaporator water flow rate
xnew_evaporator_water_flow_rate = np.linspace(EVPAORATOR_WATER_FLOW_RATE_SCALE[0][0], EVPAORATOR_WATER_FLOW_RATE_SCALE[-1][0], num=N_SAMPLES).reshape(-1,1)
scale_evaporator_water_flow_rate = evaporator_water_flow_regressor.predict(pipeline.fit_transform(xnew_evaporator_water_flow_rate)).reshape(-1)

# Scale baseline power input prediction based on condenser water scaling factor
# Examples:
# scaled_power_input[:,:,0] # power input scaled along condenser water range and cooling power range
# scaled_power_input[:,0,0] # cooling output range with minimum condenser water temp
# scaled_power_input[0,:,0] # minimum cooling output with condenser water temp range
# scaled_power_input[:,:,5] # power input scaled along cooling load required (our baseline)
scaled_power_input[:,:,0] = np.multiply(scaled_power_input[:,:,0], np.add(1,scale_condenser_water_temperature).reshape(-1,1)) # Condenser water temperature only
scaled_power_input[:,:,1] = np.multiply(scaled_power_input[:,:,1], np.add(1,scale_condenser_water_flow).reshape(-1,1)) # Condenser water flow rate
scaled_power_input[:,:,2] = np.multiply(scaled_power_input[:,:,2], np.add(1,scale_evaporator_water_return_temperature).reshape(-1,1)) # Evaporator water return temperature
scaled_power_input[:,:,3] = np.multiply(scaled_power_input[:,:,3], np.add(1,scale_evaporator_water_supply_temperature).reshape(-1,1)) # Evaporator water supply temperature
scaled_power_input[:,:,4] = np.multiply(scaled_power_input[:,:,4], np.add(1,scale_evaporator_water_flow_rate).reshape(-1,1)) # Evaporator water flow rate
assert all(scaled_power_input[0,:,5] == baseline_power_input.reshape(-1)) # This axis has not been touched yet

# Generate data for other values around the default value with noise
def condenser_water_temperature_random() -> float:
    val = random_noise_standard_normal(min=DEFAULT_CONDENSER_WATER_TEMPERATURE-1, max=DEFAULT_CONDENSER_WATER_TEMPERATURE+1)
    return val
def condenser_water_flow_rate_random() -> float:
    val = random_noise_standard_normal(min=DEFAULT_CONDENSER_WATER_FLOW_RATE-0.025, max=DEFAULT_CONDENSER_WATER_FLOW_RATE+0.025)
    return val
def evaporator_return_water_temperature_random() -> float:
    val = random_noise_standard_normal(min=DEFAULT_EVAPORATOR_RETURN_WATER_TEMPERATURE-0.5, max=DEFAULT_EVAPORATOR_RETURN_WATER_TEMPERATURE+0.5)
    return val
def evaporator_supply_water_temperature_random() -> float:
    val = random_noise_standard_normal(min=DEFAULT_EVAPORATOR_SUPPLY_WATER_TEMPERATURE-0.15, max=DEFAULT_EVAPORATOR_SUPPLY_WATER_TEMPERATURE+0.15)
    return val
def evaporator_water_flow_rate_random() -> float:
    val = random_noise_standard_normal(min=DEFAULT_EVAPORATOR_WATER_FLOW_RATE-0.025, max=DEFAULT_EVAPORATOR_WATER_FLOW_RATE+0.025)
    return val


#%% Save data to comma separated values

with open(SAVED_DATA_FILEPATH, 'wt', newline='', encoding='UTF-8') as csvfile:
    writer = csv.writer(csvfile)
    """
    Independent variables:
    * Cooling load (power) required [kW]
    * Condenser water temperature [DEG C]
    * Condenser water flow rate [percent of rated for chiller]
    * Number of operable chillers [integer]
    * Evaporator return water temperature
    * Evaporator supply water temperature (setpoint)
    * Evaporator water flow rate [percent of rated for chiller]
    Dependent variables:
    * power input
    """
    # Write file headers
    writer.writerow(FILE_HEADERS)

    # Conenser water temperature scaling
    idx_slice = 0
    for idx_output_capacity in range(scaled_power_input.shape[1]): # Scale along output cooling capacity
        for idx_independent_variable in range(scaled_power_input.shape[0]): # Scale along independent data
            data = [
                capacity_output[idx_output_capacity,0],
                scaled_power_input[idx_independent_variable, idx_output_capacity, idx_slice],
                xnew_condenser_water_temperature[idx_independent_variable,0],
                condenser_water_flow_rate_random(),
                evaporator_return_water_temperature_random(), 
                evaporator_supply_water_temperature_random(),
                evaporator_water_flow_rate_random(),
                ]
            writer.writerow(data)

    # Condenser water flow rate
    idx_slice = 1
    for idx_output_capacity in range(scaled_power_input.shape[1]): # Scale along output cooling capacity
        for idx_independent_variable in range(scaled_power_input.shape[0]): # Scale along independent data
            data = [
                capacity_output[idx_output_capacity,0],
                scaled_power_input[idx_independent_variable, idx_output_capacity, idx_slice],
                condenser_water_temperature_random(),
                xnew_condenser_water_flow_rate[idx_independent_variable,0],
                evaporator_return_water_temperature_random(), 
                evaporator_supply_water_temperature_random(),
                evaporator_water_flow_rate_random(),
                ]
            writer.writerow(data)

    # Evaporator water return temperature
    idx_slice = 2
    for idx_output_capacity in range(scaled_power_input.shape[1]): # Scale along output cooling capacity
        for idx_independent_variable in range(scaled_power_input.shape[0]): # Scale along independent data
            data = [
                capacity_output[idx_output_capacity,0],
                scaled_power_input[idx_independent_variable, idx_output_capacity, idx_slice],
                condenser_water_temperature_random(),
                condenser_water_flow_rate_random(),
                xnew_evaporator_water_return_temperature[idx_independent_variable,0], 
                evaporator_supply_water_temperature_random(),
                evaporator_water_flow_rate_random(),
                ]
            writer.writerow(data)

    # Evaporator water supply temperature
    idx_slice = 3
    for idx_output_capacity in range(scaled_power_input.shape[1]): # Scale along output cooling capacity
        for idx_independent_variable in range(scaled_power_input.shape[0]): # Scale along independent data
            data = [
                capacity_output[idx_output_capacity,0],
                scaled_power_input[idx_independent_variable, idx_output_capacity, idx_slice],
                condenser_water_temperature_random(),
                condenser_water_flow_rate_random(),
                evaporator_return_water_temperature_random(), 
                xnew_evaporator_water_supply_temperature[idx_independent_variable,0],
                evaporator_water_flow_rate_random(),
                ]
            writer.writerow(data)

    # Evaporator water flow rate
    idx_slice = 4
    for idx_output_capacity in range(scaled_power_input.shape[1]): # Scale along output cooling capacity
        for idx_independent_variable in range(scaled_power_input.shape[0]): # Scale along independent data
            data = [
                capacity_output[idx_output_capacity,0],
                scaled_power_input[idx_independent_variable, idx_output_capacity, idx_slice],
                condenser_water_temperature_random(),
                condenser_water_flow_rate_random(),
                evaporator_return_water_temperature_random(), 
                evaporator_supply_water_temperature_random(),
                xnew_evaporator_water_flow_rate[idx_independent_variable,0],
                ]
            writer.writerow(data)

# %%
