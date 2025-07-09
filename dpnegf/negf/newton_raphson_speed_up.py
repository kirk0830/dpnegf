import time
import unittest
from typing import Optional, Tuple, List, Callable
import numpy as np
from scipy.sparse import lil_array

def _impose_j_bound(inout, nx, ny, nz, typ, val, mask):
    '''impose the special mask for boundary points'''
    my_boundary_types_ = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
    my_displ_ = [+1, -1, +nx, -nx, +nx*ny, -nx*ny]
    for t, d in zip(my_boundary_types_, my_displ_):
        # perf bottle-neck: search
        i = np.where(np.array(typ) == t)[0] # all indices of the boundary type
        assert len(i) > 0, f'no grid point with type {t} found'
        # scipy sparse matrix also supports the parallized assignment
        inout[i, i] = val
        inout[i, i + d] = mask

def _impose_b_bound(inout, nx, ny, nz, typ, phi, dirichlet_pot):
    '''impose the special mask for boundary points in b vector'''
    my_boundary_types_ = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 'Dirichlet']
    my_displ_ = [+1, -1, +nx, -nx, +nx*ny, -nx*ny]
    for t, d in zip(my_boundary_types_, my_displ_):
        # perf bottle-neck: search
        i = np.where(np.array(typ) == t)[0]
        assert len(i) > 0, f'no grid point with type {t} found'
        inout[i] = phi[i] - phi[i + d]
    ipot = np.where(np.array(typ) == 'Dirichlet')[0]
    inout[ipot] = dirichlet_pot[ipot]
    inout *= -1 # for convenience change the sign of B in later NR iteration

class TestImposeBoundaryPerf(unittest.TestCase):
    def setUp(self):
        # assuming there are 1e6 grid points, each direction has 1e3
        self.nx, self.ny, self.nz = 100, 100, 100
        self.nr = self.nx * self.ny * self.nz
        # say we have 6 boundary types, each has 300 points
        typ = ['xmin'] * 30 + ['xmax'] * 30 + ['ymin'] * 30 \
            + ['ymax'] * 30 + ['zmin'] * 30 + ['zmax'] * 30
        # and all others are of type "in"
        typ += ['in'] * (self.nr - 180)
        self.typ = np.array(typ, dtype=str)
        
    def test_impose_bound(self):
        # create a dummy inout matrix
        ref = lil_array((self.nr, self.nr), dtype=float)
        nx, ny, nz = self.nx, self.ny, self.nz
        t = time.time()
        
        for i in range(self.nr):
            if self.typ[i] == "xmin":
                ref[i, i] = -1.0
                ref[i, i + 1] = 1.0
            elif self.typ[i] == "xmax":
                ref[i, i] = -1.0
                ref[i, i - 1] = 1.0
            elif self.typ[i] == "ymin":
                ref[i, i] = -1.0
                ref[i, i + nx] = 1.0
            elif self.typ[i] == "ymax":
                ref[i, i] = -1.0
                ref[i, i - nx] = 1.0
            elif self.typ[i] == "zmin":
                ref[i, i] = -1.0
                ref[i, i + nx*ny] = 1.0
            elif self.typ[i] == "zmax":
                ref[i, i] = -1.0
                ref[i, i - nx*ny] = 1.0
        print(f'old boundary impose method took {time.time() - t:.4f} seconds')
        
        inout = lil_array((self.nr, self.nr), dtype=float)
        t = time.time()
        # call the function
        _impose_j_bound(inout, nx, ny, nz, self.typ, -1.0, 1.0)
        print(f'_impose_j_bound took {time.time() - t:.4f} seconds')
    
        # check if the inout matrix is equal to the reference matrix
        self.assertTrue(all(inout[i, i] == ref[i, i] for i in range(self.nr)))
        self.assertTrue(all(inout[i, i + 1] == ref[i, i + 1] for i in range(30)))
        self.assertTrue(all(inout[i, i - 1] == ref[i, i - 1] for i in range(30, 60)))
        self.assertTrue(all(inout[i, i + nx] == ref[i, i + nx] for i in range(60, 90)))
        self.assertTrue(all(inout[i, i - nx] == ref[i, i - nx] for i in range(90, 120)))
        self.assertTrue(all(inout[i, i + nx*ny] == ref[i, i + nx*ny] for i in range(120, 150)))
        self.assertTrue(all(inout[i, i - nx*ny] == ref[i, i - nx*ny] for i in range(150, 180)))

def coloumb(chr: float|np.ndarray) -> float|np.ndarray:
    '''convert the charge to unit of Coulomb'''
    assert isinstance(chr, (float, np.ndarray)), "chr must be a float or numpy array"
    from dpnegf.utils.constants import elementary_charge as e
    return chr * e

def _jflux_impl(nx, ny, nz, r, typ, sigma, eps, feps, with_index=False):
    '''kernal implementation of the Jacobian calculation.'''

    # get all indices of the grid points of type "in"
    i = np.where(typ == "in")[0]
    # j stands for flux, <listcomp> faster than for loop
    mydisp_ = [-1, +1, -nx, +nx, -nx*ny, +nx*ny]
    jflux = [sigma[i, j // 2] * feps(np.roll(eps, d)[i], eps[i]) / np.abs(np.roll(r, d, axis=0)[i, j // 2] - r[i, j // 2])
            for j, d in enumerate(mydisp_)]
    if with_index:
        return jflux, i
    else:
        return jflux

def calculate(jinout: Optional[lil_array], 
              binout: Optional[np.ndarray],
              grid_dim: Tuple[int, int, int]|List[int],
              gridpoint_coords: np.ndarray, 
              gridpoint_typ: List[str]|np.ndarray,
              gridpoint_surfarea: np.ndarray, 
              eps: np.ndarray, 
              phi: np.ndarray, 
              phi_: np.ndarray, 
              free_chr: np.ndarray,
              fixed_chr: Optional[np.ndarray] = None,
              dirichlet_pot: Optional[np.ndarray] = None,
              eps0: float = 8.854187817e-12, 
              beta: float = 1/(1.380649e-23 * 300), 
              feps: Callable[[float, float], float] = lambda eps1, eps2: (eps1 + eps2) / 2.0) -> None:
    '''
    refactored version of the implementation to calculate the Jacobian for the Poisson equation
    using the Newton-Raphson method (finit difference method).
    
    NOTE: the input would be modified in-place
    
    Parameters
    ----------
    jinout : scipy.sparse.lil_array
        The Jacobian matrix to be calculated. It is a sparse matrix in LIL format. If is None,
        then the Jacobian will not be calculated.
    binout : np.ndarray
        The b vector to be calculated, which is the right-hand side of the linear system.
        It is a dense numpy array. If is None, then the b vector will not be calculated.
    grid_dim : tuple or list of int
        The dimensions of the grid as (nx, ny, nz), where nx, ny, and nz are the number of grid 
        points in the x, y, and z directions, respectively.
    gridpoint_coords : np.ndarray
        The coordinates of the grid points, shape (nr, 3), where nr is the total number of grid
        points.
    gridpoint_typ : list of str
        The type of grid point, with length nr
    gridpoint_surfarea : np.ndarray
        The area of Voronoi surface in three dimension, shape (nr, 3). For example the gridpoint_
        area[ir, 0] indicates the area of surface whose norm vector is along x-axis
    eps : np.ndarray
        The relative permittivity of each grid point
    phi : np.ndarray
        The electrostatic potential mapped on real-space grid point
    phi\_ : np.ndarray
        The "old" electrostatic potential mapped on real-space grid point
    free_chr : np.ndarray
        The free charge density mapped on real-space grid point
    fixed_chr : np.ndarray, optional
        The fixed charge density mapped on real-space grid point, shape (nr,). 
    dirichlet_pot : np.ndarray, optional
        The Dirichlet potential mapped on real-space grid point, shape (nr,). 
    eps0 : float, optional
        The vacuum permittivity, a constant value. Default is 8.854187817e-12 F/m.
    beta : float, optional
        The inverse temperature (1/kBT), used in the Boltzmann factor for charge density.
        Default is 1/(1.380649e-23 * 300), which corresponds to room temperature (300 K). 
    feps : callable, optional
        the functor to calculate the average permittivity on a pair of grid points.
        It should take two arguments, eps1 and eps2, and return the average permittivity.
        If not provided, it defaults to the arithmetic mean.
    '''
    # shortcut: nothing to do
    if jinout is None and binout is None:
        return
    
    # check dimensions
    if not isinstance(grid_dim, (tuple, list)) or len(grid_dim) != 3:
        raise ValueError("grid_dim must be a tuple or list of three integers")
    if any(not isinstance(d, int) or d <= 0 for d in grid_dim):
        raise ValueError("grid_dim must contain positive integers")
    nx, ny, nz = grid_dim; nr = nx * ny * nz
    
    # simple sanity check
    if jinout is not None:
        if not isinstance(jinout, lil_array):
            raise TypeError("jinout (jacobian) must be a scipy.sparse.lil_array")
        if jinout.shape != (nr, nr):
            raise ValueError(f"jinout (jacobian) must have shape ({nr}, {nr}), but got {jinout.shape}")

    if binout is not None:
        if not isinstance(binout, np.ndarray):
            raise TypeError("binout must be a numpy array")
        if binout.shape != (nr,):
            raise ValueError(f"binout must have shape ({nr},), but got {binout.shape}")
    
    # check the common input parameters
    if not isinstance(gridpoint_coords, np.ndarray) or gridpoint_coords.shape != (nr, 3):
        raise ValueError(f"gridpoint_coords must be a numpy array of shape ({nr}, 3)")
    if not isinstance(gridpoint_typ, list) or len(gridpoint_typ) != nr:
        raise ValueError(f"gridpoint_typ must be a list of length {nr}")
    if not isinstance(gridpoint_surfarea, np.ndarray) or gridpoint_surfarea.shape != (nr, 3):
        raise ValueError(f"gridpoint_surfarea must be a numpy array of shape ({nr}, 3)")
    if not isinstance(eps, np.ndarray) or eps.shape != (nr,):
        raise ValueError(f"eps must be a numpy array of shape ({nr},)")
    if not isinstance(phi, np.ndarray) or phi.shape != (nr,):
        raise ValueError(f"phi must be a numpy array of shape ({nr},)")
    if not isinstance(phi_, np.ndarray) or phi_.shape != (nr,):
        raise ValueError(f"phi_ must be a numpy array of shape ({nr},)")
    if not isinstance(free_chr, np.ndarray) or free_chr.shape != (nr,):
        raise ValueError(f"free_chr must be a numpy array of shape ({nr},)")
    
    if binout is not None:
        if not isinstance(fixed_chr, np.ndarray) or fixed_chr.shape != (nr,):
            raise ValueError(f"fixed_chr must be a numpy array of shape ({nr},)")
        if not isinstance(dirichlet_pot, np.ndarray) or dirichlet_pot.shape != (nr,):
            raise ValueError(f"dirichlet_pot must be a numpy array of shape ({nr},)")

    if not isinstance(eps0, (float, int)):
        raise TypeError("eps0 must be a float or int")
    if not isinstance(beta, (float, int)):
        raise TypeError("beta must be a float or int")
    if not callable(feps):
        raise TypeError("feps must be a callable function with strictly two arguments: "
                        "eps1 and eps2 to calculate the average permittivity")
    
    # ================================ implementation =======================================
    # making alias for more succint code
    
    r     = gridpoint_coords
    typ   = np.array(gridpoint_typ, dtype=str)
    sigma = gridpoint_surfarea
    zfree = free_chr
    zfixed = fixed_chr
    
    # calculate the flus of jacobian for all grid points of type "in"
    jflux, i = _jflux_impl(nx, ny, nz, r, typ, sigma, eps, feps, with_index=True)
    
    if jinout is not None: # only calculate jacobian when jinout is provided (allocated)
        # add flux term to matrix jinout
        jinout[i, i]  = -eps0 * np.sum(jflux, axis=0)
        jinout[i, i] -= beta * \
            coloumb(zfree[i]) * np.exp(-beta * np.sign(zfree[i]) * (phi[i] - phi_[i]))
        
        mydisp_ = [-1, +1, -nx, +nx, -nx*ny, +nx*ny]
        for d, j_ in zip(mydisp_, jflux):
            jinout[i, i + d] = eps0 * j_
        _impose_j_bound(jinout, nx, ny, nz, typ, val=1.0, mask=-1.0)

    if binout is not None: # only calculate b vector when binout is provided (allocated)
        bflux = [jf * (phi[i + d] - phi[i]) for jf, d in zip(jflux, mydisp_)]
        binout[i]  = np.sum(bflux, axis=0)
        binout[i] += coloumb(zfree[i]) * np.exp(-beta * np.sign(zfree[i]) * (phi[i] - phi_[i]))
        binout[i] += coloumb(zfixed[i])

        _impose_b_bound(binout, nx, ny, nz, typ, phi, dirichlet_pot)

if __name__ == '__main__':
    unittest.main()
