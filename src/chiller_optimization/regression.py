"""Given a set of data, create a regression model which estimates the target
variable (power input) given a set of independent variables

Notes on loading and saving models within this module
Loading a model should be possible with only the customer ID. Model file names are comprised of a 
customer ID, description of the model, and time stamp"""
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
from abc import ABCMeta
from glob import glob
from uuid import UUID

# Third party imports
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
import numpy as np

# Local imports
from .pipeline import linear_regression_pipeline
from .data_load import load_training_data_csv, REQUIRED_FILE_HEADERS
from . import parser

# Declaration
DEGREE: int = int(parser['pipeline']['degree_of_polynomial_features'])
ALPHA: float = float(parser['hyperparameters']['ridge_alpha'])
DATA_FILEPATH = '../../data/generated_dummy_data2022-9-11.csv'
TARGET_HEADER_NAME = 'power_input [kW]'

# Saving a model
MODEL_BASE_FILENAME: str = '{customer_id}_{model_description}_{datetime}'
MODEL_TYPE: Dict[str,str] = {
    str(type(LinearRegression())): 'linear_regression',
    str(type(Ridge())): 'ridge_regression',
    str(type(SGDRegressor())): 'SGD_regression',
}
LINEAR_PARAMETERS_BASE_FILENAME = '{customer_id}_linear_parameters_{datetime}'
DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%S"

#%%

def load_model_parameters(customer_id: str) -> Union[LinearRegression, Ridge]:
    """TODO - deferred because this function isn't planned to be needed yet
    Load a saved model based on the customer ID
    This is intended for future use when models will be loaded
    when a customer requests a prediction. 
    An alternative to returning an pickled model is to load the 
    weights associated with a model and initialize the model with
    the trained weights... unknown what to do right now"""
    raise NotImplementedError
    return None

def save_model_parameters(
    customer_id: str,
    save_directory: str,
    model: Union[LinearRegression, Ridge, SGDRegressor]) -> str:
    """Save a models parameters, but not the entire model"""
    model_filename: str = determine_model_file_name(customer_id, model)
    model_filepath: str = os.path.join(save_directory, model_filename)

    # TODO
    # Currently, I do not see a benefit of saving model parameters separately.
    # I do not plan on implementing this right now
    raise NotImplementedError

def load_model_pickled(
    customer_id: str,
    save_directory: str,) -> Union[LinearRegression, Ridge, SGDRegressor]:
    """
    inputs
    -------
    customer_id (str): Unique ID string or UUID related to a customer
    Determine the most recently saved model associated with a customer ID
    then load the model"""

    model: Any
    # Determine the most recent saved model
    model_path_match: str = os.path.join(save_directory, customer_id)
    models = glob(model_path_match + '*')
    _sort_strings_on_date_format(models)
    # Read the disc file to memory
    with open(models[0]) as model_file:
        model = pickle.load(model_file)
    return model

# Save a model
def save_model_pickled(
        customer_id: str, 
        save_directory: str, 
        model: Union[LinearRegression, Ridge, SGDRegressor]) -> str:
    """Save a trained model associated with a customer"""
    model_filename: str = determine_model_file_name(customer_id, model)
    model_filepath: str = os.path.join(save_directory, model_filename)

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

    return model_filepath




def _sort_strings_on_date_format(string_list: List[str]) -> None:
    """Return a sorted list of strings in descending order, with the sorting
    function using a datetime string representation"""
    string_list.sort(key = lambda x: datetime.strptime(x[-19:], DATETIME_FORMAT))
    return None

def determine_model_file_name(
    customer_id: str, 
    model: Union[LinearRegression, Ridge, SGDRegressor]) -> str:
    """Save a model in a specified location with a specified file name"""
    if isinstance(model, ABCMeta):
        msg="Model is not initialized. Got {}, expected {}".format(type(model), type(LinearRegression()))
        raise ValueError(msg)

    model_file_name:str
    model_description: str = MODEL_TYPE[str(type(model))]
    datetime_str:str = datetime.now(tz=timezone.utc).strftime(DATETIME_FORMAT)
    
    model_file_name = MODEL_BASE_FILENAME.format(
        customer_id=customer_id,
        datetime=datetime_str,
        model_description=model_description,
    )
    return model_file_name

def determine_linear_model_parameters_file_name(
    customer_id: str) -> str:
    """Save parameters related to a model in a specified location with a specified file name
    This will ONLY determine the file name for linear model parameters. The whole model is
    not saved, only the weights and bias"""
    if not isinstance(customer_id, int):
        raise ValueError("Customer ID should be integer, not {}".format(type(customer_id)))

    parameters_file_name:str
    datetime_str:str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%HH:MM:SS")
    
    parameters_file_name = LINEAR_PARAMETERS_BASE_FILENAME.format(
        customer_id=customer_id,
        datetime=datetime_str,
    )
    return parameters_file_name


