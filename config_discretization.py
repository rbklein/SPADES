"""
    Configure discretization options and parameters

    Discretization details:
        - cell-centered finite volume method
"""

import jax

#Datatypes
set_DTYPE = "DOUBLE"

match set_DTYPE:
    case "DOUBLE":
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        DTYPE = jnp.float64
    case "SINGLE":
        import jax.numpy as jnp
        DTYPE = jnp.float32


#Gravity coefficient
g           = 3

#Mesh [-length,length]
length      = 1
num_cells   = 300
dx          = 2 * length / num_cells
x           = jnp.linspace(-length + 0.5 * dx, length - 0.5 * dx, num_cells, dtype = DTYPE)

#Temporal
time_final  = 1.0
num_steps   = 2000
dt          = time_final / num_steps

#Numerics
TEST_CASE   = "DAM_BREAK"
FLUX        = "FJORDHOLM"
LIMITER     = "MINMOD"
DISSIPATION = "TECNO_ROE"
INTEGRATOR  = "TADMOR"