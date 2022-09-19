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
from typing import List

# Third party imports
import numpy as np

# Local imports

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


#%%

def load_training_data_csv(filepath: str, headers:List[str]) -> np.ndarray:
    """Read .csv file to list of tuples"""
    rows: List[float] = []

    with open(filepath, 'rt', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile)
        headers: List[str] = next(reader)
        if not set(headers).issubset(set(REQUIRED_FILE_HEADERS)):
            msg=f'Improperly named .csv headers. Expected {REQUIRED_FILE_HEADERS}. Got {headers}'
            raise ValueError(msg)

        for row in reader:
            rows.append(row)

    return np.array(rows, dtype=np.float32)

#%%