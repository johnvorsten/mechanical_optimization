"""Pipeline defined for regression and data modeling

TODO - cleaning of data
missing data uses interpolation between two values

configuration file = config.ini
Location of data to load
number of features expected
degree of polynomial features

"""

# Python imports
from configparser import ConfigParser

# Third party imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Local imports

# Declarations
CONFIG_FILEPATH = './config.ini'
parser = ConfigParser()
parser.read(CONFIG_FILEPATH)
DEGREE: int = int(parser['pipeline']['degree_of_polynomial_features'])

#%%

linear_regression_pipeline = Pipeline(
    [
    ('polynomial', PolynomialFeatures(degree=DEGREE)),
    ('scaler', StandardScaler()),
    ]
)