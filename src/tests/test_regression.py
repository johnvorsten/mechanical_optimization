"""Tests for regression.py"""

#%%
# Python imports
import unittest
from typing import List
from datetime import datetime, timezone
from uuid import UUID, uuid4
from copy import deepcopy
import os

# Third party imports
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split

# Local imports
from chiller_optimization.data_load import (
    load_training_data_csv, REQUIRED_FILE_HEADERS)
from chiller_optimization.regression import (
    DATA_FILEPATH,
    determine_linear_model_parameters_file_name, 
    determine_model_file_name,
    load_model_pickled, save_model_pickled,
    _sort_strings_on_date_format,
    DATETIME_FORMAT)
from chiller_optimization.pipeline import linear_regression_pipeline

# Declarations
CUSTOMER_ID = '9756f3c5-b316-4bff-9a4f-c0ac78c7f4f5'
SAVE_DIRECTORY = './'

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

    def test__sort_strings_on_date_format(self):

        # Generate a few example strings based on template
        examples: List[str] = [
            '9756f3c5-b316-4bff-9a4f-c0ac78c7f4f5_linear_regression_2022-10-15T14:29:51',
            '3eed37ab-9e88-4dc6-98b4-892c8dc79e72_linear_regression_2022-10-15T14:29:52',
            '18432f89-9d77-4e1c-87dc-77a6346ded01_linear_regression_2022-10-15T14:29:53',
            '0a8e0697-2bfc-4311-b426-99705e104e14_linear_regression_2022-10-15T14:28:51',
            '2630cfc3-d1f4-4e5a-a993-9af0441c5a07_linear_regression_2022-10-15T13:29:51'
            ]
        sorted_examples: List[str] = [
            '2630cfc3-d1f4-4e5a-a993-9af0441c5a07_linear_regression_2022-10-15T13:29:51',
            '0a8e0697-2bfc-4311-b426-99705e104e14_linear_regression_2022-10-15T14:28:51',
            '9756f3c5-b316-4bff-9a4f-c0ac78c7f4f5_linear_regression_2022-10-15T14:29:51',
            '3eed37ab-9e88-4dc6-98b4-892c8dc79e72_linear_regression_2022-10-15T14:29:52',
            '18432f89-9d77-4e1c-87dc-77a6346ded01_linear_regression_2022-10-15T14:29:53'
            ]
        examples.sort(key = lambda x: datetime.strptime(x[-19:], DATETIME_FORMAT))
        self.assertEqual(examples, sorted_examples)

        # Shuffle examples to test function
        examples.append(examples.pop(0))
        _sort_strings_on_date_format(examples)
        self.assertEqual(examples, sorted_examples)

        return None

    def test_load_model_pickled(self):
        """Create a dummy model which is saved to disc. Attempt to save and load the model to disc"""

        # Create a dummy model
        # Load data
        DATA_FILEPATH = '../../data/generated_dummy_data2022-9-11.csv'
        data = load_training_data_csv(DATA_FILEPATH, required_headers=REQUIRED_FILE_HEADERS)
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
        ridge_model = Ridge(fit_intercept=True, alpha=1)
        # Data pipeline
        X = linear_regression_pipeline.fit_transform(features)
        # Split data into training and testing slices
        # Shuffle data loaded
        x_train, x_test, y_train, y_test= train_test_split(X, target, test_size=0.15, shuffle=True)
        # Define a model
        linear_model = LinearRegression(fit_intercept=True)
        ridge_model = Ridge(fit_intercept=True, alpha=1)
        linear_model.fit(x_train, y_train)
        ridge_model.fit(x_train, y_train)

        # Save model to disc
        model_filepath_linear = save_model_pickled(CUSTOMER_ID, SAVE_DIRECTORY, linear_model)

        # Read model from disc
        model_linear = load_model_pickled(CUSTOMER_ID, SAVE_DIRECTORY)
        
        # Attempt to use the model
        model_linear.predict(x_test)

        # Save model to disc
        model_filepath_ridge = save_model_pickled(CUSTOMER_ID, SAVE_DIRECTORY, ridge_model)

        # Read model from disc
        model_ridge = load_model_pickled(CUSTOMER_ID, SAVE_DIRECTORY)
        
        # Attempt to use the model
        model_ridge.predict(x_test)

        # Delete old models from disc
        os.remove(model_filepath_ridge)
        os.remove(model_filepath_linear)

        return None

    def test_save_model_pickled(self):
        """save_model_pickled must return the model filepath and exist as a file."""
        # Define a model
        linear_model = LinearRegression(fit_intercept=True)
        ridge_model = Ridge(fit_intercept=True, alpha=1)

        # Save model to disc
        model_filepath_linear = save_model_pickled(CUSTOMER_ID, SAVE_DIRECTORY, linear_model)

        # Read model from disc
        model_linear = load_model_pickled(CUSTOMER_ID, SAVE_DIRECTORY)
        
        # Save model to disc
        model_filepath_ridge = save_model_pickled(CUSTOMER_ID, SAVE_DIRECTORY, ridge_model)

        # Read model from disc
        model_ridge = load_model_pickled(CUSTOMER_ID, SAVE_DIRECTORY)

        # Delete old models from disc
        os.remove(model_filepath_ridge)
        os.remove(model_filepath_linear)

        return None
    

if __name__ == '__main__':
    unittest.main()
# %%
