"""
    Configure ROM options and parameters
"""

from config_discretization import *

#Reduced space
num_modes = 200

#Data
file_name = "snapshot_data.npy"

basis_params = (num_modes, file_name)

#Numerics
BASIS       = "STANDARD"
PROJECTION  = "GALERKIN"
METHOD      = "STANDARD"
INTEGRATOR  = "RK4"
