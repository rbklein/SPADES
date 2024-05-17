"""
    Contains numerical time integrators
"""

from functools import partial

from config_discretization import *

import entropy
import discrete_gradient
import minimization

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

@partial(jax.jit, static_argnums = 1)
def Tadmor_midpoint(u, dudt):
    """
        Locally entropy conservative Crank-Nicolsen time integrator by Tadmor with Gonzalez discrete gradient "Entropy stability theory for difference 
        approximations of nonlinear conservation laws and related time-dependent problems"
    """

    #minimization requires vector data but u is matrix
    u_guess         = jnp.reshape(u, 2 * num_cells)

    #evaluates entropy average between vectorized new state and the old state u
    average_u       = lambda u_new: entropy.conservative_variables(discrete_gradient.Gonzalez(jnp.reshape(u_new, (2,-1)), u))

    #evaluates FOM residual at Gonzalez entropy average
    residual_Crank_Nicolson = lambda u_new: jnp.reshape(jnp.reshape(u_new, (2,-1)) - u - dt * dudt(average_u(u_new)), (2 * num_cells))

    u_new = minimization.levenberg_marquardt(residual_Crank_Nicolson, u_guess, 1e-6, 20)

    #reshape new state back to matrix
    return jnp.reshape(u_new, (2,-1))

def return_integrator(which_integrator):
    """
        Returns the specific numerical time integration function that will be used for computations

        Options:
            - RK4
            - Modified Crank-Nicolson
    """
    match which_integrator:
        case "RK4":
            """
                The classical 4-th order Runge-Kutta time integrator
            """
            integrator = RK4
        case "TADMOR":
            """
                The locally entropy conservative modified Crank-Nicolson method by Tadmor
            """
            integrator = Tadmor_midpoint

    return integrator

if __name__ == "__main__":
    def rosenbrock(x):
        return jnp.array([10 * (x[1] - x[0]**2), (1 - x[0])])

    gn = LevenbergMarquardt(residual_fun=rosenbrock)
    gn_sol = gn.run(jnp.zeros(2, dtype=DTYPE)).params

    print(gn_sol)