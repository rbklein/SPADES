"""
    Compute initial conditions and specify functions based on configurations
"""

from config_ROM import *
from setup import *

import basis
import integration

basis_generator = basis.return_basis_generator(BASIS)
coefficients    = basis.return_coefficient_generator(BASIS)
integrator      = integration.return_ROM_integrator(INTEGRATOR)

Phi_0   = basis_generator(basis_params)
a_0     = coefficients(Phi_0, u_0)
