
"""
    Compute initial conditions and specify functions based on configurations

    variables:
        - h:    water column height
        - hu:   discharge
"""

from config_discretization import *

import testsuite
import flux
import dissipation
import integrator

initial_h, initial_hu = testsuite.return_case(TEST_CASE)

f_cons  = flux.return_flux(FLUX)
f_diss  = dissipation.return_dissipation(DISSIPATION)
lim     = dissipation.return_limiter(LIMITER)
step    = integrator.return_integrator(INTEGRATOR)

h_0     = initial_h(x)
hu_0    = initial_hu(x)

u_0     = jnp.array([h_0, hu_0], dtype=DTYPE)

