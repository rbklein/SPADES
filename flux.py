"""
    Contains functions to compute entropy conservative numerical fluxes
"""

from config_discretization import *

#shift value for jnp.roll
shift = -1

@jax.jit
def a_mean(quantity):
    """
        Computes arithmetic mean of quantity 

        The i-th component is the mean between cell i and i+1
    """
    return 0.5 * (quantity + jnp.roll(quantity, shift))

@jax.jit
def Fjordholm_flux(u):
    """
        Computes flux of Fjordholm, Mishra and Tadmor from "Energy preserving and energy stable schemes
        for the shallow water equations" 
    """
    h_mean      = a_mean(u[0])
    vel         = u[1] / u[0]
    vel_mean    = a_mean(vel)

    h_squared_mean = a_mean(u[0]**2)

    F1 = h_mean * vel_mean
    F2 = h_mean * vel_mean**2 + g/2 * h_squared_mean
    return jnp.array([F1,F2], dtype=DTYPE)


def return_flux(which_flux):
    """
        Returns the specific flux function that will be used for computations

        Options:
            - Fjordholm, Mishra, Tadmor
    """
    match which_flux:
        case "FJORDHOLM":
            """
                The flux by Fjordholm, Mishra and Tadmor
            """
            flux = Fjordholm_flux

    return flux