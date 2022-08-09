"""
Create B-splines which represent dummy equipment operating curves
"""
#%%
# Python imports

# Third party imports
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
import numpy as np

# Local imports

# Declarations
# Interpolations
data_points = np.array([ # condenser water temperature, power input, capacity output
    [35, 21.9, 49.2], # maximum capacity, COP 2.246
    [35, 6.84, 9.84], # Minimum capacity, COP 1.43
    [35, 9.37, 34.4], # highest efficiency, COP 3.67
    [35, 7.46, 19.4], # other point, COP 2.8
    ], dtype=np.float32)
independent_data = data_points[:,[0,2]]
target = data_points[:,1]

# File import
DUMMY_DATA_FILEPATH = '../../data/dummy_data.csv'
DUMMY_DATA = np.genfromtxt(DUMMY_DATA_FILEPATH, delimiter=',')
chiller_1_data = DUMMY_DATA[:, [1,2,5]]
chiller_2_data = DUMMY_DATA[:, [1,3,6]]
chiller_3_data = DUMMY_DATA[:, [1,4,7]]

#%%



# %%
