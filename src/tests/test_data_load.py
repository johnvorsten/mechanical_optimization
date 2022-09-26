# Python imports
import unittest
from typing import List
import os

# Third party imports

# Local imports
from chiller_optimization.data_load import load_training_data_csv

# Declarations
REQUIRED_FILE_HEADERS: List[str] = [
    'capacity_output [kW]',
    'power_input [kW]',
    'condenser_water_temperature [DEG C]',
    'condenser_water_flow_rate [percent]',
    'evaporator_water_return_temperature [DEG C]',
    'evaporator_water_supply_temperature [DEG C]',
    'evaporator_water_flow_rate [percent]',
    ]
DUMMY_DATA_FILENAME = '../../data/generated_dummy_data2022-9-11.csv'
dirname = os.path.dirname(__file__)
DUMMY_DATA_FILEPATH = os.path.join(dirname, DUMMY_DATA_FILENAME)

#%%

class DataInputTest(unittest.TestCase):

    def setUp(self):
        return None

    def test_load_training_data_csv(self):

        load_training_data_csv(DUMMY_DATA_FILEPATH, REQUIRED_FILE_HEADERS)
        
        with self.assertRaises(ValueError):
            load_training_data_csv(DUMMY_DATA_FILEPATH, ['headers','not','right'])

        return None

if __name__ == '__main__':
    unittest.main()