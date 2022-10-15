# Python imports
import unittest
from typing import List
from datetime import datetime, timezone

# Third party imports
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor


# Local imports
from chiller_optimization.data_load import load_training_data_csv
from chiller_optimization.regression import (
    determine_linear_model_parameters_file_name, 
    determine_model_file_name, load_model,
    load_model_pickled, save_model_parameters, 
    save_model_pickled)

# Declarations
CUSTOMER_ID = 1


#%%

class RegressionTest(unittest.TestCase):

    def setUp(self):
        return None
    
    def test_determine_linear_model_parameters_file_name(self):
        """"""
        parameters_filename = determine_linear_model_parameters_file_name(CUSTOMER_ID)

        # Parse datetime
        parts = parameters_filename.split('_')

        # constant file name portion
        self.assertEqual(parts[1], 'linear')
        self.assertEqual(parts[2], 'parameters')

        # Test customer ID with the standard method
        self.assertEqual(int(parts[0]), CUSTOMER_ID)
        self.assertIsInstance(parts[0], str)

        # Test that datetime is UTC, and not some local time
        utc_datetime = datetime.utcnow().strftime("%Y-%m-%dT%HH:MM:SS") # Not timezone aware, but this is OK
        utc_datetime2 = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%HH:MM:SS") # timezone aware
        desired_datetime:str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%HH:MM:SS")
        self.assertEqual(parts[3], utc_datetime)
        self.assertEqual(parts[3], utc_datetime2)
        self.assertEqual(parts[3], desired_datetime)

        return None
    
    def test_determine_model_file_name(self):
        """"""
        model = LinearRegression()
        model_filename = determine_model_file_name(CUSTOMER_ID, model)

        # Parse datetime
        parts = model_filename.split('_')

        # constant file name portion
        self.assertEqual(parts[1], 'linear')
        self.assertEqual(parts[2], 'regression')

        # Test customer ID with the standard method
        self.assertEqual(int(parts[0]), CUSTOMER_ID)
        self.assertIsInstance(parts[0], str)

        # Test that datetime is UTC, and not some local time
        utc_datetime = datetime.utcnow().strftime("%Y-%m-%dT%HH:MM:SS") # Not timezone aware, but this is OK
        utc_datetime2 = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%HH:MM:SS") # timezone aware
        desired_datetime:str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%HH:MM:SS")
        self.assertEqual(parts[3], utc_datetime)
        self.assertEqual(parts[3], utc_datetime2)
        self.assertEqual(parts[3], desired_datetime)

        # Different model type, Ridge
        model = SGDRegressor()
        model_filename = determine_model_file_name(CUSTOMER_ID, model)

        # Parse datetime
        parts = model_filename.split('_')

        # constant file name portion
        self.assertEqual(parts[1], 'SGD')
        self.assertEqual(parts[2], 'regression')

        # Test customer ID with the standard method
        self.assertEqual(int(parts[0]), CUSTOMER_ID)
        self.assertIsInstance(parts[0], str)

        # Different model type, Ridge
        model = Ridge()
        model_filename = determine_model_file_name(CUSTOMER_ID, model)

        # Parse datetime
        parts = model_filename.split('_')

        # constant file name portion
        self.assertEqual(parts[1], 'ridge')
        self.assertEqual(parts[2], 'regression')

        # Test customer ID with the standard method
        self.assertEqual(int(parts[0]), CUSTOMER_ID)
        self.assertIsInstance(parts[0], str)
        return None

    def test_save_model_pickled(self):
        raise NotImplementedError
        return None

    def test_load_model_pickled(self):
        raise NotImplementedError
        return None

    def test_(self):
        return None

    def test_(self):
        return None

    def test_(self):
        return None

    def test_(self):
        return None
    

if __name__ == '__main__':
    unittest.main()