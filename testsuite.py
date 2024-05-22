"""
    A testsuite containing initial conditions for different test cases
"""

from config_discretization import *

def return_case(test_case):
    """
        Returns initial conditions for test case given by test_case
    """
    match test_case:
        case "DAM_BREAK":
            initial_h = lambda x: jnp.where(jnp.abs(x) < 0.2, 1.5, 1)     
            initial_hu = lambda x: x * 0.0 
        case "DROP":
            initial_h = lambda x: 0.1 * jnp.exp(-100 * x**2) + 1
            initial_hu = lambda x: x * 0 
        case "ENTROPY_RIEMANN":
            initial_h = lambda x: jnp.where(jnp.abs(x) < 12.5, 1.5, 0.02)     
            initial_hu = lambda x: x * 0.0 



    return initial_h, initial_hu