"""
    Contains numerical time integrators
"""

from functools import partial

from config_discretization import *

@partial(jax.jit, static_argnums = 1)
def RK4(u, dudt):
    """
        The classical 4-th order Runge-Kutta time integrator
    """
    k1 = dudt(u)
    k2 = dudt(u + k1 * dt / 2)
    k3 = dudt(u + k2 * dt / 2)
    k4 = dudt(u + k3 * dt)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def return_integrator(which_integrator):
    """
        Returns the specific numerical time integration function that will be used for computations

        Options:
            - RK4
    """
    match which_integrator:
        case "RK4":
            """
                The classical 4-th order Runge-Kutta time integrator
            """
            integrator = RK4

    return integrator
