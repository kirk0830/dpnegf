import numpy as np
from scipy.sparse import lil_array, coo_matrix

def calc_jac(jac, 
             nx, 
             ny, 
             nz, 
             grid_xyz, 
             grid_typ,
             grid_s_voro, 
             eps, 
             phi, 
             phi_, 
             free_chr,
             eps0, 
             beta, 
             e,
             feps=lambda eps1, eps2: (eps1 + eps2) / 2):
    '''
    refactored version of the implementation to calculate the Jacobian for the Poisson equation
    using the Newton-Raphson method (finit difference method).
    
    NOTE: the input would be modified in-place
    
    Parameters
    ----------
    jac : scipy.sparse.lil_array
        The Jacobian matrix to be calculated. It is a sparse matrix in LIL format.
    feps : callable
        the functor to calculate the average permittivity on a pair of grid points.
        It should take two arguments, eps1 and eps2, and return the average permittivity.
        If not provided, it defaults to the arithmetic mean.
    '''
    # simple sanity check
    if not isinstance(jac, lil_array):
        raise TypeError("jac must be a scipy.sparse.lil_array")
    if not callable(feps):
        raise TypeError("feps must be a callable function")
    nr = nx * ny * nz
    if jac.shape != (nr, nr):
        raise ValueError(f"jac must have shape ({nr}, {nr}), but got {jac.shape}")
    
    for gp_index in range(nr):
        if grid_typ[gp_index] == "in":
            flux_xm_J = grid_s_voro[gp_index,0]*eps0*feps(eps[gp_index-1],eps[gp_index])\
            /abs(grid_xyz[gp_index,0]-grid_xyz[gp_index-1,0])

            flux_xp_J = grid_s_voro[gp_index,0]*eps0*feps(eps[gp_index+1],eps[gp_index])\
            /abs(grid_xyz[gp_index+1,0]-grid_xyz[gp_index,0])
            
            flux_ym_J = grid_s_voro[gp_index,1]*eps0*feps(eps[gp_index-nx],eps[gp_index])\
            /abs(grid_xyz[gp_index-nx,1]-grid_xyz[gp_index,1])

            flux_yp_J = grid_s_voro[gp_index,1]*eps0*feps(eps[gp_index+nx],eps[gp_index])\
            /abs(grid_xyz[gp_index+nx,1]-grid_xyz[gp_index,1])

            flux_zm_J = grid_s_voro[gp_index,2]*eps0*feps(eps[gp_index-nx*ny],eps[gp_index])\
            /abs(grid_xyz[gp_index-nx*ny,2]-grid_xyz[gp_index,2])

            flux_zp_J = grid_s_voro[gp_index,2]*eps0*feps(eps[gp_index+nx*ny],eps[gp_index])\
            /abs(grid_xyz[gp_index+nx*ny,2]-grid_xyz[gp_index,2])

            # add flux term to matrix jac
            jac[gp_index,gp_index] = -(flux_xm_J+flux_xp_J+flux_ym_J+flux_yp_J+flux_zm_J+flux_zp_J)\
                +e*free_chr[gp_index]*(-np.sign(free_chr[gp_index])) * beta*\
                np.exp(-np.sign(free_chr[gp_index])*(phi[gp_index]-phi_[gp_index]) * beta)
            jac[gp_index,gp_index-1] = flux_xm_J
            jac[gp_index,gp_index+1] = flux_xp_J
            jac[gp_index,gp_index-nx] = flux_ym_J
            jac[gp_index,gp_index+nx] = flux_yp_J
            jac[gp_index,gp_index-nx*ny] = flux_zm_J
            jac[gp_index,gp_index+nx*ny] = flux_zp_J

        else: # boundary points
            jac[gp_index,gp_index] = 1.0 # correct for both Dirichlet and Neumann boundary condition
            
            if grid_typ[gp_index] == "xmin":   
                jac[gp_index,gp_index+1] = -1.0
            elif grid_typ[gp_index] == "xmax":
                jac[gp_index,gp_index-1] = -1.0
            elif grid_typ[gp_index] == "ymin":
                jac[gp_index,gp_index+nx] = -1.0
            elif grid_typ[gp_index] == "ymax":
                jac[gp_index,gp_index-nx] = -1.0
            elif grid_typ[gp_index] == "zmin":
                jac[gp_index,gp_index+nx*ny] = -1.0
            elif grid_typ[gp_index] == "zmax":
                jac[gp_index,gp_index-nx*ny] = -1.0
