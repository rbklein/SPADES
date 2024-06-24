"""
    Generate ROM
"""
from config_discretization import *
from config_ROM import *
from setup_ROM import *

import projection
import integration

def generate_method(which_projection, which_integrator, which_method):
    """
        Generate ROM method
    """
    match(which_projection):
        case "LSPG":
            residual = projection.generate_discrete_residual(which_integrator)
            method = projection.generate_LSPG_method(which_method)

            @jax.jit
            def step(a, Phi):
                return method(a, a, Phi, residual)
        
        case "GALERKIN":
            Galerkin_projection = projection.generate_Galerkin_method(which_method)
            
            @jax.jit
            def step(a, Phi):
                return integrator(a, Phi, Galerkin_projection)

    return step

step = generate_method(PROJECTION, INTEGRATOR, METHOD)