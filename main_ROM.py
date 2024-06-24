"""
    Main file for reduced order computation containing a JAX-based solution algorithm of the shallow water equations
"""

from setup import *
from setup_ROM import *

from computational import lift

import ROM
import plot

Phi     = jnp.copy(Phi_0)
a       = jnp.copy(a_0)

time_index = 0
while time_index < num_steps:
    a = ROM.step(a, Phi)

    time_index += 1
    print(time_index * dt)

plot.plot_all(x, lift(Phi, a))
plot.show()
