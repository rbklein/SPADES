"""
    Combines all element of the spatial discretization i.e.:
        - entropy conservative flux
        - entropy dissipation operator
        - entropy dissipation limiter
"""

from config_discretization import *
from setup import *

@jax.jit
def dudt(u):
    """
        Returns the ODEs corresponding to the semi-discrete shallow water equations using the specified 
        entropy conservative flux, entropy dissipation operator and limiter
    """
    F = f_cons(u)
    D = f_diss(u, lim)
    return - ((F - jnp.roll(F,shift=1,axis=1)) + (D - jnp.roll(D,shift=1,axis=1)) ) / dx

