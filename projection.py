"""
    Contains function to compute the projection of a full order model
"""

from config_discretization import *
from config_ROM import *

from functools import partial
from computational import lift, project, vectorize_data

import discrete_gradient
import FOM
import minimization
import entropy

@jax.jit
def Tadmor_residual(a_new, a_old, Phi):
    """
        Computes fully discrete ROM residual using method adapted from Tadmor "Entropy stability theory for difference 
        approximations of nonlinear conservation laws and related time-dependent problems"

        Output is vectorized for minimization algorithms
    """
    u_old = lift(Phi, a_old)
    u_new = lift(Phi, a_new)

    u_average = entropy.conservative_variables(discrete_gradient.Gonzalez(u_new, u_old))

    return vectorize_data(u_new - u_old - dt * FOM.dudt(u_average))

def generate_discrete_residual(which_integrator):
    """
        Generates residual function for LSPG ROM
    """
    match(which_integrator):
        case "TADMOR":
            """
                residual based on method adapted from Tadmor 
            """
            residual = Tadmor_residual

    return residual

@partial(jax.jit, static_argnums = (3))
def LSPG_minimize(a_guess, a_old, Phi, residual):
    """
        Minimizes a given residual for a ROM solution
    """
    minimization_residual = lambda a: residual(a, a_old, Phi)

    return minimization.levenberg_marquardt(minimization_residual, a_guess)

@partial(jax.jit, static_argnums = (3))
def ES_LSPG_minimize(a_guess, a_old, Phi, residual):
    """
        Minimizes a given residual for a ROM solution with entopy stability constraints
    """
    pass

@partial(jax.jit, static_argnums = (3))
def P_ES_LSPG_minimize(a_guess, a_old, Phi, residual):
    """
        Minimizes a given residual for a ROM solution with positivity preserving and entopy stability constraints
    """
    pass

def generate_LSPG_method(which_LSPG):
    """
        Generate LSPG solver

        Options:
            - standard LSPG method
            - entropy stable LSPG method
            - positivity preserving and entropy stable LSPG method
    """
    match(which_LSPG):
        case "STANDARD":
            """
                Standard LSPG method
            """
            LSPG = LSPG_minimize
        case "ES_LSPG":
            """
                Entropy stable LSPG method
            """
            LSPG = ES_LSPG_minimize
        case "P_ES_LSPG":
            """
                Positivity preserving and entropy stable LSPG method
            """
            LSPG = P_ES_LSPG_minimize

    return LSPG

@jax.jit
def Galerkin_projection(a, Phi):
    """
        Compute dadt using standard Galerkin projection method
    """
    return project(Phi, FOM.dudt(lift(Phi, a)))

@jax.jit
def Entropy_projected_Galerkin(a, Phi):
    """
        Compute dadt using Galerkin projection with entropy projection method
    """
    eta = entropy.entropy_variables(lift(Phi, a))
    projected_eta = lift(Phi, project(Phi, eta))
    return project(Phi, FOM.dudt(entropy.conservative_variables(projected_eta)))
    
def generate_Galerkin_method(which_Galerkin):
    """
        Generate Galerkin solver

        Options:
            - standard Galerkin
            - entropy projected Galerkin
    """    
    match(which_Galerkin):
        case "STANDARD":
            """
                Standard Galerkin projection
            """
            Galerkin = Galerkin_projection
        case "ENTROPY_PROJECTION":
            """
                Entropy projected Galerkin
            """
            Galerkin = Entropy_projected_Galerkin

    return Galerkin