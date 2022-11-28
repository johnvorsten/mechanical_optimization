"""Given a set of data, create a regression model which estimates the target
variable (power input) given a set of independent variables

Notes on loading and saving models within this module
Loading a model should be possible with only the customer ID. Model file names are comprised of a 
customer ID, description of the model, and time stamp"""
#%%

# Python imports
from configparser import ConfigParser
from typing import Union, Any, List, Dict, Tuple
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
from chiller_optimization.pipeline import linear_regression_pipeline
from chiller_optimization import parser

# Declaration
DEGREE: int = int(parser['pipeline']['degree_of_polynomial_features'])
ALPHA: float = float(parser['hyperparameters']['ridge_alpha'])
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
    if len(models) == 0:
        msg=('The directory does not contain any files with ' +
             'names matching the desired path match string {model_path_match}')
        raise OSError(msg)
    # Read the disc file to memory
    with open(models[0], 'rb') as model_file:
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

def delete_model_pickled(customer_id: str, save_directory: str, model_created_when: str) -> str:
    """Delete either the most recent model (newest), or the oldest model
    inputs
    -------
    customer_id: unique ID related to customer
    save_directory: (str) directory wehere model is saved
    model_created_when: (str) one of ['oldest', 'newest']
    """
    model: Any
    # Final all model filepaths related to a specific customer
    model_path_match: str = os.path.join(save_directory, customer_id)
    models = glob(model_path_match + '*')
    _sort_strings_on_date_format(models) # descending
    if len(models) == 0:
        msg=('The directory does not contain any files with ' +
             'names matching the desired path match string {model_path_match}')
        raise OSError(msg)
    
    # Determine which model to look at
    if len(models) == 1:
        msg=("There is only one model currently saved. No models will be deleted " +
        "if there is only one model available.")

    if model_created_when == 'newest':
        model_filepath = model[0]
    elif model_created_when == 'oldest':
        model_filepath = model[-1]
    else:
        msg=('Incorrect value of model_created_when. Must be one of {"newest", "oldest"}')
        raise ValueError(msg)

    # Delete model from disc
    os.remove(model_filepath)

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

