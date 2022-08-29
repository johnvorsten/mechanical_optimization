"""
Create B-splines which represent dummy equipment operating curves
"""
#%%
# Python imports
from typing import Tuple
import csv

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

#%%
xs = np.linspace(start=9, stop=60, num=50)
ys = interpolator(np.stack(
            (np.array([35]*50, dtype=np.float32), xs), axis=1)
            )
efficiency = xs / ys
fig, ax = plt.subplots()
ax_efficiency = ax.twinx()
ax.scatter(xs, ys)
ax_efficiency.plot(xs, efficiency, c='red', label='efficiency')
ax.axvline(x=9.84, dashes=(5,5))
ax.axvline(x=49.2, dashes=(5,5))
ax.text(x=9.84, y=35, s='minimum\ncapacity')
ax.text(x=49.2, y=35, s='maximum\ncapacity')
ax.set_xlabel('Capacity output (kW of cooling)')
ax.set_ylabel('Electric power input (kW)')
ax_efficiency.set_ylabel('Efficiency [kWA/kW]')
ax.set_title('Interpolation of cooling capacity output ')
ax_efficiency.legend(loc='lower center')
plt.show()


# %%
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

#%%

def random_noise(min:float, max:float, size:Tuple[int,int]) -> float:
    return (max - min) * rng.random(size) + min

def random_noise_standard_normal(min:float, max:float, size:Tuple[int,int]) ->float:
    return (max - min) * rng.standard_normal(size) + min

pipeline = Pipeline(
    [
    ('polynomial', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ]
)

# Scale the dependent data based on our weights to create dummy data
desired_capacity = np.array(data_points[:,2]).reshape(-1,1) # Capacity output (desired output capacity)
dependent_data = data_points[:,1] # Power input (given the desired output capacity, power input is predicted)
linear_regressor_power_input = LinearRegression(fit_intercept=True, n_jobs=2)
X = pipeline.fit_transform(desired_capacity)
y = dependent_data
linear_regressor_power_input.fit(X, y)

N_SAMPLES = 50
x_scale = np.linspace(start=9, stop=60, num=N_SAMPLES).reshape(-1,1) # Cooling power output [kW]
y_hat = linear_regressor_power_input.predict(pipeline.fit_transform(x_scale)) # Predicted power input [kW]

efficiency = np.divide(x_scale.reshape(-1), y_hat)
fig, ax = plt.subplots()
ax_efficiency = ax.twinx()
ax.scatter(x_scale, y_hat)
ax_efficiency.plot(xs, efficiency, c='red', label='efficiency')
ax.axvline(x=9.84, dashes=(5,5))
ax.axvline(x=49.2, dashes=(5,5))
ax.text(x=9.84, y=35, s='minimum\ncapacity')
ax.text(x=49.2, y=35, s='maximum\ncapacity')
ax.set_xlabel('Capacity output (kW of cooling)')
ax.set_ylabel('Electric power input (kW)')
ax_efficiency.set_ylabel('Efficiency [kWA/kW]')
ax.set_title('Interpolation of cooling capacity output ')
ax_efficiency.legend(loc='lower center')
plt.show()


#%%

# For each independent variable, scale the dependent data and save the generated data
N_SAMPLES = 50
independent_data = np.array([val for (val, _) in CONDENSER_WATER_TEMPERATURE_SCALE]).reshape(-1,1)
y = np.array([scale for (_, scale) in CONDENSER_WATER_TEMPERATURE_SCALE]).reshape(-1,1)
condenser_water_temperature_regressor = LinearRegression(fit_intercept=True, n_jobs=2)
X = pipeline.fit_transform(independent_data)
condenser_water_temperature_regressor.fit(X, y)

# Create data for prediction
xnew = np.linspace(CONDENSER_WATER_TEMPERATURE_SCALE[0][0], CONDENSER_WATER_TEMPERATURE_SCALE[-1][0], num=N_SAMPLES).reshape(-1,1) # Condenser water temperature
ynew = condenser_water_temperature_regressor.predict(pipeline.fit_transform(xnew)) # Power scale percent
capacity_output = x_scale = np.linspace(start=9.9, stop=49, num=N_SAMPLES).reshape(-1,1) # (50,1) power output [kW]
scaled_power_input = linear_regressor_power_input.predict(capacity_output) # TODO left off

# Generate data for other values around the default value with noise
condenser_water_flow_rate = (
    np.array([DEFAULT_CONDENSER_WATER_FLOW_RATE]*N_SAMPLES) + 
    random_noise_standard_normal(min=DEFAULT_CONDENSER_WATER_FLOW_RATE-0.025, max=DEFAULT_CONDENSER_WATER_FLOW_RATE+0.025, size=(N_SAMPLES))
)
evaporator_return_water_temperature = (
    np.array([DEFAULT_CONDENSER_WATER_TEMPERATURE]*N_SAMPLES) + 
    random_noise_standard_normal(min=DEFAULT_CONDENSER_WATER_TEMPERATURE-2, max=DEFAULT_CONDENSER_WATER_TEMPERATURE+2, size=(N_SAMPLES))
)
evaporator_supply_water_temperature = (
    np.array([DEFAULT_EVAPORATOR_SUPPLY_WATER_TEMPERATURE]*N_SAMPLES) + 
    random_noise_standard_normal(min=DEFAULT_EVAPORATOR_SUPPLY_WATER_TEMPERATURE-0.25, max=DEFAULT_EVAPORATOR_SUPPLY_WATER_TEMPERATURE+0.25, size=(N_SAMPLES))
)
evaporator_water_flow_rate = (
    np.array([DEFAULT_EVAPORATOR_WATER_FLOW_RATE]*N_SAMPLES) + 
    random_noise_standard_normal(min=DEFAULT_EVAPORATOR_WATER_FLOW_RATE-0.25, max=DEFAULT_EVAPORATOR_WATER_FLOW_RATE+0.25, size=(N_SAMPLES))
)

fig, ax = plt.subplots()
ax.plot(xnew, ynew, '--')
ax.legend(['Result of power input'], loc='best')
ax.set_title("Example of condenser water temperature\non power input required")
ax.set_xlabel("Condenser water temperature [DEG C]")
ax.set_ylabel("Power input scale\nscaled_power_input = power_input * (1+y) [%]")
plt.show()

with open(SAVED_DATA_FILEPATH, 'wt', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for idx in range(N_SAMPLES):
        """
        Independent variables:
        * Cooling load (power) required [kW]
        * Condenser water temperature [DEG C]
        * Condenser water flow rate [percent of rated for chiller]
        * Number of operable chillers [integer]
        * Evaporator return water temperature
        * Evaporator supply water temperature (setpoint)
        * Evaporator water flow rate [percent of rated for chiller]
        """
        data = [
            cooling_capacity_output
            scaled_power_input[idx],
            evaporator_return_temperature, 
            evaporator_supply_temperature,
            evaporator_flow_rate,
            condenser_flow_rate,

        ]
        writer.writerow()
# %%
