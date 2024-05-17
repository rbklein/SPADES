"""
    Contains discrete gradient functions
"""

from config_discretization import *

import entropy


@jax.jit
def Gonzalez(u1,u2):
    """
        Computes the Gonzalez discrete gradient appearing in paper: "Time integration and discrete Hamiltonian systems"
    """
    u_mean = 0.5 * (u1 + u2)
    u_jump = u1 - u2
    s_jump = entropy.entropy(u1) - entropy.entropy(u2)

    #computes squared norm at each point in grid
    u_norm_squared = jnp.sum(u_jump**2, axis=0)

    #entropy variables at arithmetic average of states
    eta_mean    = entropy.entropy_variables(u_mean)

    #Gonzalez correction factor
    factor      = (s_jump - jnp.sum(eta_mean * u_jump, axis = 0)) / u_norm_squared
    factor      = jnp.nan_to_num(factor, copy=False, nan=0)

    #use broadcasting to multiply every component of u_jump by factor
    eta_gonzalez = eta_mean + factor[None,:] * u_jump

    return eta_gonzalez

