"""
    Contains function to compute entropy dissipation operators to use with entropy conservative fluxes for the shallow water equations
"""
from functools import partial

from config_discretization import *

import entropy
import flux

@jax.jit
def jump_vec(quantity):
    """
        Compute jump in vector-valued quantity on grid 

        The i-th component is the jump between cells i+1 and i

        indices:
            quantity: 0 row index, 1 grid index
    """
    return jnp.roll(quantity, shift=-1, axis=1) - quantity

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
def minmod(delta1, delta2):
    """
        Computes minmod limiter values

        Suitable for vector-valued inputes
    """
    #if delta2 is more than zero return delta2 else return a very small value
    delta3 = jnp.where(jnp.abs(delta2) > 0, delta2, 1e-14)
    phi = jnp.where(delta1 / delta3 < 0, 0, delta1 / delta3)
    return jnp.where(phi < 1, phi, 1)

def return_limiter(which_limiter):
    """
        Returns the specific limiter function that will be used for computations of the entropy dissipation operator

        Options:
            - Minmod
    """
    match which_limiter:
        case "MINMOD":
            """
                The minmod limiter
            """
            limiter = minmod

    return limiter

@partial(jax.jit, static_argnums = 1)
def first_order(u, limiter, u_ref = 0):
    """
        Computes a first-order accuracte entropy dissipation operator

        To be implemented
    """
    pass

@partial(jax.jit, static_argnums = 1)
def Roe_dissipation(u, limiter, u_ref = 0):
    """
        Computes second-order Roe entropy dissipation operator from the paper of Fjordholm, Mishra and Tadmor from "Arbitrarily High-order 
        Accurate Entropy Stable Essentially Nonoscillatory Schemes for Systems of Conservation Laws" using limiter
    """
    h_mean      = flux.a_mean(u[0])
    hu_mean     = flux.a_mean(u[1])
    vel_mean    = hu_mean / h_mean
    vel_char    = jnp.sqrt(g * h_mean)

    eta         = entropy.entropy_variables(u)
    eta_jump    = jump_vec(eta)

    #compute eigenvector at mean state
    R = jnp.ones((2,2,num_cells))
    R = R.at[1,0,:].set(vel_mean - vel_char); R = R.at[1,1,:].set(vel_mean + vel_char)

    #compute eigenvalue matrices at mean state
    D = jnp.zeros((2,2,num_cells))
    D = D.at[0,0,:].set(jnp.abs(vel_mean - vel_char)); D = D.at[1,1,:].set(jnp.abs(vel_mean + vel_char))

    #compute inner product of entropy variable jump and eigenvector basis
    delta = mul(R.transpose(1,0,2), eta_jump)

    limiter_mean = 0.5 * (limiter(jnp.roll(delta, shift=1, axis=1), delta) + limiter(jnp.roll(delta, shift=-1, axis=1), delta))

    #compute scaling coefficient eigenvalues
    S = jnp.zeros((2,2,num_cells))
    S = S.at[0,0,:].set(1 - limiter_mean[0,:]); S = S.at[1,1,:].set(1 - limiter_mean[1,:])

    return -mul(R, mul(D, mul(S, delta))) * 1/2

@partial(jax.jit, static_argnums = 1)
def frozen_Roe_dissipation(u, limiter, u_ref):
    """
        Computes Roe entropy dissipation operator from the paper of Fjordholm, Mishra and Tadmor at frozen state u_ref "Arbitrarily High-order 
        Accurate Entropy Stable Essentially Nonoscillatory Schemes for Systems of Conservation Laws" using limiter
    """
    h_mean      = flux.a_mean(u_ref[0])
    hu_mean     = flux.a_mean(u_ref[1])
    vel_mean    = hu_mean / h_mean
    vel_char    = jnp.sqrt(g * h_mean)

    eta_ref     = entropy.entropy_variables(u_ref)
    eta_ref_jump= jump_vec(eta_ref)

    eta         = entropy.entropy_variables(u)
    eta_jump    = jump_vec(eta)

    #compute eigenvector at mean state
    R = jnp.ones((2,2,num_cells))
    R = R.at[1,0,:].set(vel_mean - vel_char); R = R.at[1,1,:].set(vel_mean + vel_char)

    #compute eigenvalue matrices at mean state
    D = jnp.zeros((2,2,num_cells))
    D = D.at[0,0,:].set(jnp.abs(vel_mean - vel_char)); D = D.at[1,1,:].set(jnp.abs(vel_mean + vel_char))

    #compute inner product of entropy variable jump and eigenvector basis
    delta_ref = mul(R.transpose(1,0,2), eta_ref_jump)
    delta = mul(R.transpose(1,0,2), eta_jump)

    limiter_mean = 0.5 * (limiter(jnp.roll(delta_ref, shift=1, axis=1), delta_ref) + limiter(jnp.roll(delta_ref, shift=-1, axis=1), delta_ref))

    #compute scaling coefficient eigenvalues
    S = jnp.zeros((2,2,num_cells))
    S = S.at[0,0,:].set(1 - limiter_mean[0,:]); S = S.at[1,1,:].set(1 - limiter_mean[1,:])

    return -mul(R, mul(D, mul(S, delta))) * 1/2

def return_dissipation(which_dissipation):
    """
        Returns the specific dissipation operator that will be used for computations of entropy stable numerical fluxes

        Options:
            - Roe dissipation
            - frozen Roe dissipation
            - Laplacian (not implemented)
            - No dissipation
    """
    match which_dissipation:
        case "TECNO_ROE":
            """
                Roe-based dissipation operator from the TeCNO class schemes
            """
            dissipation = Roe_dissipation
        case "FROZEN_ROE":
            """
                Roe-based dissipation operator evaluated a frozen state
            """
            dissipation = frozen_Roe_dissipation
        case "LAPLACIAN":
            """
                Simple Laplacian dissipation
            """
            raise NotImplementedError("Laplacian dissipation has not been implemented yet")
        case "NONE":
            """
                No dissipation
            """
            dissipation = lambda u, lim, u_ref: 0.0 * u

    return dissipation
