"""Given a set of data, create a regression model which estimates the target
variable (power input) given a set of independent variables"""
#%%

# Python imports
from configparser import ConfigParser
from typing import Union, Any, List, Dict
from datetime import datetime, timezone
import time
import pickle
import os
import csv
from copy import deepcopy

# Third party imports
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_validate
from matplotlib import pyplot as plt
from sklearn.metrics import (explained_variance_score, mean_absolute_error, mean_squared_error, max_error)

# Local imports
from pipeline import linear_regression_pipeline
from data_load import load_training_data_csv, REQUIRED_FILE_HEADERS

# Declaration
parser = ConfigParser()
parser.read('./config.ini')
DEGREE: int = int(parser['pipeline']['degree_of_polynomial_features'])
ALPHA: float = float(parser['hyperparameters']['ridge_alpha'])
DATA_FILEPATH = '../../data/generated_dummy_data2022-9-11.csv'
TARGET_HEADER_NAME = 'power_input [kW]'

# Saving a model
MODEL_BASE_FILENAME = '{customer_id}_{model_type}_{datetime}'
MODEL_TYPE = {
    str(type(LinearRegression)): 'linear_regression',
    str(type(Ridge)): 'ridge_regression',
    str(type(SGDRegressor)): 'SGD_regression',
}


#%%

# Load a model
def load_model(customer_id: int) -> Union[LinearRegression, Ridge]:
    """TODO
    Load a saved model based on the customer ID
    This is intended for future use when models will be loaded
    when a customer requests a prediction. 
    An alternative to returning an pickled model is to load the 
    weights associated with a model and initialize the model with
    the trained weights... unkonwn what to do right now"""
    return None

def load_model_pickled(
    customer_id: int,
    save_directory: str,) -> Union[LinearRegression, Ridge, SGDRegressor]:
    """Determine the most recently saved model associated with a customer ID
    then load the model"""

    model: Any
    # TODO
    # Determine the most recent saved model
    
    # Read the disc file to memory
    # Load the memory into a representation of a model (most likely scikit learn)

    return model

# Save a model
def save_model_pickled(
        customer_id: int, 
        save_directory: str, 
        model: Union[LinearRegression, Ridge, SGDRegressor]) -> str:
    """TODO - save a trained model associated with a customer"""
    model_filename: str = determine_model_file_name(customer_id, model)
    model_filepath: str = os.path.join(save_directory, model_filename)

    with open(model_filepath, 'wb') as file:
        pickle.dump(model)

    return model_filepath

def determine_model_file_name(
    customer_id: int, 
    model: Union[LinearRegression, Ridge, SGDRegressor]) -> str:
    """Save a model in a specificed location with a specified file name"""
    model_file_name:str
    model_description: str = MODEL_TYPE[str(type(model))]
    datetime_str:str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%HH:MM:SS")
    
    model_file_name = MODEL_BASE_FILENAME.format(
        customer_id=customer_id,
        datetime=datetime_str,
        model_description=model_description,
    )
    return model_file_name


