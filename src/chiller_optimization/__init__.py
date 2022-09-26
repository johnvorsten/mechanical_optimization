
# Python imports
import os
from configparser import ConfigParser

# Third party imports

# Local imports

# Declarations
parser = ConfigParser()
CONFIG_FILENAME = 'config.ini'
CONFIG_FILEPATH = os.path.join(__path__[0], CONFIG_FILENAME)
parser.read(CONFIG_FILEPATH)