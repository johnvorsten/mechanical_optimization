"""Read data from a structured source
Initially this will be a .csv file. In the future it will possibly
be a database, or remote file storage. For now, simply create a flat
interface through with data can be loaded. Future methods can be added
for alternate data sources.

Load the data into a structured format for passing to other methods later.
Options: dictionary, named tuple, list, array
Choose array for easy processing later

configuration file = config.ini
Location of data to load
number of features expected

"""
#%%

# Python imports
import csv
from typing import List, Tuple
from copy import deepcopy

# Third party imports
import numpy as np

# Local imports
from chiller_optimization.regression import LINEAR_INPUT_SPECIFICATION

# Declaration
REQUIRED_FILE_HEADERS: List[str] = [
    'capacity_output [kW]',
    'power_input [kW]',
    'condenser_water_temperature [DEG C]',
    'condenser_water_flow_rate [percent]',
    'evaporator_water_return_temperature [DEG C]',
    'evaporator_water_supply_temperature [DEG C]',
    'evaporator_water_flow_rate [percent]',
    ]
FEATURE_INDEX: List[int] = list(range(len(REQUIRED_FILE_HEADERS))).remove(REQUIRED_FILE_HEADERS.index('power_input [kW]'))
TARGET_INDEX: List[int] = REQUIRED_FILE_HEADERS.index('power_input [kW]')
DUMMY_DATA_FILEPATH = '../data/generated_dummy_data2022-9-11.csv' # Relative to module

#%%

def load_training_data_csv(filepath: str, required_headers:List[str]) -> np.ndarray:
    """Read .csv file to list of tuples"""
    rows: List[float] = []

    with open(filepath, 'rt', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile)
        headers: List[str] = next(reader)
        if not set(required_headers).issubset(set(REQUIRED_FILE_HEADERS)):
            msg=f'Improperly named .csv headers. Expected {REQUIRED_FILE_HEADERS}. Got {headers}'
            raise ValueError(msg)

        for row in reader:
            rows.append(row)

    return np.array(rows, dtype=np.float32)

def load_dummy_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Returns a tuple of (features: np.ndarray, target, feature names) related to the 
    dummy dataset included with this package"""

    data = load_training_data_csv(DUMMY_DATA_FILEPATH, required_headers=REQUIRED_FILE_HEADERS)
    features_headers: List[str] = deepcopy(REQUIRED_FILE_HEADERS)
    features_headers.remove('power_input [kW]')
    features_index: List[int] = list(range(len(REQUIRED_FILE_HEADERS)))
    features_index.remove(REQUIRED_FILE_HEADERS.index('power_input [kW]'))
    features = data[:,features_index]
    target_index = REQUIRED_FILE_HEADERS.index('power_input [kW]')
    target = data[:, target_index]

    data = (features, target, features_headers)

    return data


#%%