"""
    Contains functions to compute reduced order model basis
"""

from config_ROM import *
from computational import project, vectorize_data, unvectorize_data

def generate_standard_basis(params):
    """
        Generates a standard truncated POD basis
    """
    num_modes, file_name = params

    data = jnp.load(file_name)
    data = vectorize_data(data)

    Phi, _, _   = jnp.linalg.svd(data, full_matrices = False)
    Phi         = Phi[:,:num_modes]

    return unvectorize_data(Phi)

def return_basis_generator(which_basis):
    """
        Return a basis generator function

        Options:
            - Standard POD basis
            - Temporally localized basis
            - Completely localized basis
    """
    match(which_basis):
        case "STANDARD":
            """
                Standard POD basis
            """
            basis_generator = generate_standard_basis
        case "TEMPORAL_LOCALIZATION":
            pass
        case "COMPLETE_LOCALIZATION":
            pass
    return basis_generator

@jax.jit
def standard_coefficients(Phi, u):
    """
        Computes coefficients in standard basis
    """
    return project(Phi, u)

def return_coefficient_generator(which_basis):
    """
        Return a basis coefficient generator function

        Options:
            - Standard POD basis
            - Temporally localized basis
            - Completely localized basis
    """
    match(which_basis):
        case "STANDARD":
            """
                Standard POD basis
            """
            basis_generator = standard_coefficients
        case "TEMPORAL_LOCALIZATION":
            pass
        case "COMPLETE_LOCALIZATION":
            pass
    return basis_generator

if __name__ == "__main__":
    from computational import project, lift

    params = (200, "snapshot_data.npy")

    data = jnp.load(params[1])
    Phi = generate_standard_basis(params)
    u = data[:,:,1000]

    a = project(Phi, u)

    print(a.shape)

    import matplotlib.pyplot as plt

    plt.plot(lift(Phi, a)[0,:])
    plt.show()

