"""
    Contains frequently recurring mathematical opererations
"""

from config_discretization import *
from config_ROM import *

@jax.jit
def arith_mean(quantity):
    """
        Computes arithmetic mean of quantity 

        The mean is taken between as many cells as possible without exceeding array dimensions
    """

    return 0.5 * (quantity[1:] + quantity[:-1])

@jax.jit
def jump_vec(quantity):
    """
        Compute jump in vector-valued quantity on grid 

        Computes as many jumps as possible given the quantity

        indices:
            quantity: 0 row index, 1 grid index
    """
    return quantity[:,1:] - quantity[:,:-1]

@jax.jit
def norm2_nodal(u):
    """
        Computes the squared norm of a vector in every mesh point
    """
    return jnp.sum(u**2, axis = 0)

@jax.jit
def inner_nodal(u,v):
    """
        Computes the inner product of two vectors in every mesh point
    """
    return jnp.sum(u*v, axis = 0)

@jax.jit
def norm2(u):
    """
        Computes the squared norm of a vector on the whole grid
    """
    return jnp.sum(norm2_nodal(u))

@jax.jit
def inner(u,v):
    """
        Computes inner product on the whole grid
    """
    return jnp.sum(inner_nodal(u,v))

@jax.jit
def jump(quantity):
    """
        compute jump in scalar-valued quantity on grid
    """
    return quantity[1:] - quantity[:-1]

@jax.jit
def lift(Phi, a):
    """
        Lifts POD coefficients back to FOM space

        Take (r) and (component index, grid index, r) and outputs (component index, grid index)
    """
    return jnp.array([Phi[0,:,:] @ a, Phi[1,:,:] @ a], dtype=DTYPE)

project = jax.jit(jax.vmap(inner, (2, None), (0)))
project.__doc__ = "Projects vector u on POD basis"

@jax.jit
def mul(A,v):
    """
        Computes matrix vector product at each node of the grid

        numpy matmul docs: 'If either argument is N-Dimensional, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.'

        indices:
            A: 0 row index, 1 column index, 2 grid index
            v: 0 vector component, 1 grid index + (a dummy index to enable use of jnp.matmul)
    """
    return jnp.matmul(A.transpose((2,0,1)), v[:,:,None].transpose((1,0,2)))[:,:,0].transpose()

@jax.jit
def vectorize_data(u):
    """
        Takes a 2D or 3D array assumed to be shaped like ((data index), component index, grid index) and reshapes it to ((data index), vectorized index)
    """

    if (jnp.ndim(u) == 2):
        return jnp.reshape(u, 2 * num_cells)
    else:
        num_data = u.shape[2]
        return jnp.reshape(u, (2*num_cells, num_data))
    
@jax.jit
def unvectorize_data(u):
    """
        Takes vectorized data and projects it on grid
    """

    if (jnp.ndim(u) == 1):
        return jnp.reshape(u, (2, num_cells))
    else:
        return jnp.reshape(u, (2, num_cells, -1))