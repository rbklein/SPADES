"""
    Contains functions related to the PDE entropy of the shallow water equations
"""

from config_discretization import *

@jax.jit
def entropy(u):
    """
        Computes the PDE entropy of the shallow water equations
    """
    return 0.5 * (u[1]**2 / u[0] + g * u[0]**2)

@jax.jit
def entropy_variables(u):
    """
        Computes the entropy variables from the conserved variables of the shallow water equations
    """
    eta1 = g * u[0] - 0.5 * (u[1] / u[0])**2
    eta2 = u[1] / u[0]
    return jnp.array([eta1, eta2], dtype=DTYPE)