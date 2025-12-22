import time
import unittest

import numpy as np
from scipy.sparse import lil_matrix

from scipy.constants import epsilon_0, elementary_charge

from dpnegf.negf.newton_raphson_speed_up import (
    _impose_j_bound,
    _impose_b_bound,
    _bflux_impl,
    _jflux_impl,
    nr_construct,
)

# eps0 in units of F/Angstrom (same as poisson_init.py)
eps0 = epsilon_0 * 1e-10


class TestImposeJacobianBoundaryPerf(unittest.TestCase):
    def setUp(self):
        # assuming there are 1e6 grid points, each direction has 1e2
        self.nx, self.ny, self.nz = 100, 100, 100
        self.nr = self.nx * self.ny * self.nz
        # say we have 6 boundary types, each has 300 points
        typ = np.array(['****'] * self.nr, dtype=str).reshape((self.nx, self.ny, self.nz))
        typ[typ == '****'] = 'in'
        typ[   0,    :,  :] = 'xmin'
        typ[  -1,    :,  :] = 'xmax'
        typ[1:-1,    0,  :] = 'ymin'
        typ[1:-1,   -1,  :] = 'ymax'
        typ[1:-1, 1:-1,  0] = 'zmin'
        typ[1:-1, 1:-1, -1] = 'zmax'
        self.typ = typ.flatten()

    def test_impose_bound(self):
        # create a dummy inout matrix
        ref = lil_matrix((self.nr, self.nr), dtype=float)
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

        inout = lil_matrix((self.nr, self.nr), dtype=float)
        t = time.time()
        # call the function
        _impose_j_bound(inout, nx, ny, nz, self.typ, -1.0, 1.0)
        print(f'_impose_j_bound took {time.time() - t:.4f} seconds')

        # check if the inout matrix is equal to the reference matrix
        # check all diagonal elements
        self.assertTrue(all(inout[i, i] == ref[i, i] for i in range(self.nr)))
        # check xmin points
        ind_xmin = np.where(self.typ == 'xmin')[0]
        self.assertTrue(all(inout[i, i + 1] == ref[i, i + 1] for i in ind_xmin))
        # check xmax points
        ind_xmax = np.where(self.typ == 'xmax')[0]
        self.assertTrue(all(inout[i, i - 1] == ref[i, i - 1] for i in ind_xmax))
        # check ymin points
        ind_ymin = np.where(self.typ == 'ymin')[0]
        self.assertTrue(all(inout[i, i + nx] == ref[i, i + nx] for i in ind_ymin))
        # check ymax points
        ind_ymax = np.where(self.typ == 'ymax')[0]
        self.assertTrue(all(inout[i, i - nx] == ref[i, i - nx] for i in ind_ymax))
        # check zmin points
        ind_zmin = np.where(self.typ == 'zmin')[0]
        self.assertTrue(all(inout[i, i + nx*ny] == ref[i, i + nx*ny] for i in ind_zmin))
        # check zmax points
        ind_zmax = np.where(self.typ == 'zmax')[0]
        self.assertTrue(all(inout[i, i - nx*ny] == ref[i, i - nx*ny] for i in ind_zmax))


class TestImposeRHSVecBoundaryPerf(unittest.TestCase):
    def setUp(self):
        # assuming there are 1e6 grid points, each direction has 1e2
        self.nx, self.ny, self.nz = 100, 100, 100
        self.nr = self.nx * self.ny * self.nz
        typ = np.array(['****'] * self.nr, dtype=str).reshape((self.nx, self.ny, self.nz))
        typ[typ == '****'] = 'in'
        typ[   0,    :,  :] = 'xmin'
        typ[  -1,    :,  :] = 'xmax'
        typ[1:-1,    0,  :] = 'ymin'
        typ[1:-1,   -1,  :] = 'ymax'
        typ[1:-1, 1:-1,  0] = 'zmin'
        typ[1:-1, 1:-1, -1] = 'zmax'
        self.typ = typ.flatten()
        self.phi = np.random.rand(self.nr).astype(float)
        self.fixed_pot = np.random.rand(self.nr).astype(float)

    def test_impose_bound(self):
        # create a dummy inout vector
        ref = np.zeros(self.nr, dtype=float)
        nx, ny, nz = self.nx, self.ny, self.nz
        t = time.time()

        for i in range(self.nr):
            if self.typ[i] == "xmin":
                ref[i] = self.phi[i] - self.phi[i + 1]
            elif self.typ[i] == "xmax":
                ref[i] = self.phi[i] - self.phi[i - 1]
            elif self.typ[i] == "ymin":
                ref[i] = self.phi[i] - self.phi[i + nx]
            elif self.typ[i] == "ymax":
                ref[i] = self.phi[i] - self.phi[i - nx]
            elif self.typ[i] == "zmin":
                ref[i] = self.phi[i] - self.phi[i + nx*ny]
            elif self.typ[i] == "zmax":
                ref[i] = self.phi[i] - self.phi[i - nx*ny]
            elif self.typ[i] == "Dirichlet":
                ref[i] = self.phi[i] - self.fixed_pot[i]
        print(f'old boundary impose method took {time.time() - t:.4f} seconds')

        inout = np.zeros(self.nr, dtype=float)
        t = time.time()
        # call the function
        _impose_b_bound(inout, nx, ny, nz, self.typ, self.phi, self.fixed_pot)
        print(f'_impose_b_bound took {time.time() - t:.4f} seconds')

        # check if the inout vector is equal to the reference vector
        self.assertTrue(np.all(inout == ref))


class TestBFluxImplPerf(unittest.TestCase):
    def setUp(self):
        # assuming there are 1e6 grid points, each direction has 1e2
        self.nx, self.ny, self.nz = 100, 100, 100
        self.nr = self.nx * self.ny * self.nz
        # set the boundary types
        typ = np.array(['****'] * self.nr, dtype=str).reshape((self.nx, self.ny, self.nz))
        typ[typ == '****'] = 'in'
        typ[   0,    :,  :] = 'xmin'
        typ[  -1,    :,  :] = 'xmax'
        typ[1:-1,    0,  :] = 'ymin'
        typ[1:-1,   -1,  :] = 'ymax'
        typ[1:-1, 1:-1,  0] = 'zmin'
        typ[1:-1, 1:-1, -1] = 'zmax'
        self.typ = typ.flatten()
        self.r = np.random.rand(self.nr, 3).astype(float)
        self.sigma = np.random.rand(self.nr, 3).astype(float)
        self.eps = np.random.rand(self.nr).astype(float)
        self.eps0 = 8.854187817e-12
        self.avgeps = lambda eps1, eps2: (eps1 + eps2) / 2.0  # arithmetic mean

    def test_bflux_impl(self):
        ind = np.where(self.typ == "in")[0]
        jflux = np.random.rand(6, self.nr).astype(float)
        phi = np.random.rand(self.nr).astype(float)
        # calculate the reference result
        ref = np.zeros((6, self.nr), dtype=float)
        nx, ny, nz = self.nx, self.ny, self.nz
        t = time.time()
        for i in range(self.nr):
            if self.typ[i] == 'in':
                ref[0, i] = jflux[0, i] * (phi[i     - 1] - phi[i])
                ref[1, i] = jflux[1, i] * (phi[i     + 1] - phi[i])
                ref[2, i] = jflux[2, i] * (phi[i    - nx] - phi[i])
                ref[3, i] = jflux[3, i] * (phi[i    + nx] - phi[i])
                ref[4, i] = jflux[4, i] * (phi[i - nx*ny] - phi[i])
                ref[5, i] = jflux[5, i] * (phi[i + nx*ny] - phi[i])
        print(f'old bflux_impl method took {time.time() - t:.4f} seconds')
        # call the function
        t = time.time()
        bflux = _bflux_impl(jflux, ind, self.nx, self.ny, self.nz, phi, full_size=True)
        print(f'_bflux_impl took {time.time() - t:.4f} seconds')
        # check if the result is equal to the reference result
        self.assertTrue(bflux.shape == (6, self.nr))
        self.assertTrue(np.allclose(bflux, ref, rtol=1e-5, atol=1e-8))


class TestJFluxImplPerf(unittest.TestCase):
    def setUp(self):
        # assuming there are 1e6 grid points, each direction has 1e2
        self.nx, self.ny, self.nz = 100, 100, 100
        self.nr = self.nx * self.ny * self.nz
        # set the boundary types
        typ = np.array(['****'] * self.nr, dtype=str).reshape((self.nx, self.ny, self.nz))
        typ[typ == '****'] = 'in'
        typ[   0,    :,  :] = 'xmin'
        typ[  -1,    :,  :] = 'xmax'
        typ[1:-1,    0,  :] = 'ymin'
        typ[1:-1,   -1,  :] = 'ymax'
        typ[1:-1, 1:-1,  0] = 'zmin'
        typ[1:-1, 1:-1, -1] = 'zmax'
        self.typ = typ.flatten()
        self.r = np.random.rand(self.nr, 3).astype(float)
        self.sigma = np.random.rand(self.nr, 3).astype(float)
        self.eps = np.random.rand(self.nr).astype(float)
        self.eps0 = 8.854187817e-12  # vacuum permittivity
        self.avgeps = lambda eps1, eps2: (eps1 + eps2) / 2.0  # arithmetic mean

    def test_jflux_impl(self):
        # calculate the reference result
        ref = np.zeros((6, self.nr), dtype=float)
        nx, ny, nz = self.nx, self.ny, self.nz
        t = time.time()
        for i in range(self.nr):
            if self.typ[i] == "in":
                ref[0, i] = self.sigma[i, 0] * self.eps0 * self.avgeps(self.eps[i - 1], self.eps[i]) \
                            / np.abs(self.r[i - 1, 0] - self.r[i, 0])
                ref[1, i] = self.sigma[i, 0] * self.eps0 * self.avgeps(self.eps[i + 1], self.eps[i]) \
                            / np.abs(self.r[i + 1, 0] - self.r[i, 0])
                ref[2, i] = self.sigma[i, 1] * self.eps0 * self.avgeps(self.eps[i - nx], self.eps[i]) \
                            / np.abs(self.r[i - nx, 1] - self.r[i, 1])
                ref[3, i] = self.sigma[i, 1] * self.eps0 * self.avgeps(self.eps[i + nx], self.eps[i]) \
                            / np.abs(self.r[i + nx, 1] - self.r[i, 1])
                ref[4, i] = self.sigma[i, 2] * self.eps0 * self.avgeps(self.eps[i - nx*ny], self.eps[i]) \
                            / np.abs(self.r[i - nx*ny, 2] - self.r[i, 2])
                ref[5, i] = self.sigma[i, 2] * self.eps0 * self.avgeps(self.eps[i + nx*ny], self.eps[i]) \
                            / np.abs(self.r[i + nx*ny, 2] - self.r[i, 2])
        print(f'old jflux_impl method took {time.time() - t:.4f} seconds')
        # call the function
        t = time.time()
        jflux, i = _jflux_impl(self.nx, self.ny, self.nz, self.r, self.typ,
                               self.sigma, self.eps, self.eps0, self.avgeps,
                               with_index=True, full_size=True)
        print(f'_jflux_impl took {time.time() - t:.4f} seconds')
        # check if the result is equal to the reference result
        self.assertTrue(jflux.shape == (6, self.nr))
        self.assertTrue(i.shape == (len(self.typ[self.typ == "in"]),))
        self.assertTrue(np.allclose(jflux, ref, rtol=1e-5, atol=1e-8))


class TestNRConstructConsistency(unittest.TestCase):
    """
    Test consistency between new nr_construct implementation and the old
    implementation from poisson_init.py (NR_construct_Jac_B).

    The old implementation is reproduced here based on the commented-out code
    in poisson_init.py for direct comparison.
    """

    def setUp(self):
        # Use a smaller grid for faster testing
        self.nx, self.ny, self.nz = 10, 10, 10
        self.nr = self.nx * self.ny * self.nz

        # Set up boundary types
        # Use Fortran order (x varies fastest) to match grid coordinate indexing
        typ = np.array(['****'] * self.nr, dtype=str).reshape((self.nx, self.ny, self.nz))
        typ[typ == '****'] = 'in'
        typ[   0,    :,  :] = 'xmin'
        typ[  -1,    :,  :] = 'xmax'
        typ[1:-1,    0,  :] = 'ymin'
        typ[1:-1,   -1,  :] = 'ymax'
        typ[1:-1, 1:-1,  0] = 'zmin'
        typ[1:-1, 1:-1, -1] = 'zmax'
        self.typ = typ.flatten(order='F')

        # Set up grid coordinates (ensuring no zero distances)
        # Grid indexing: i+1 is x neighbor, i+nx is y neighbor, i+nx*ny is z neighbor
        # So we need x to vary fastest when flattening (Fortran order)
        x = np.linspace(0, 1, self.nx)
        y = np.linspace(0, 1, self.ny)
        z = np.linspace(0, 1, self.nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        # Use Fortran order to make x vary fastest, then y, then z
        self.grid_coord = np.stack([
            xx.flatten(order='F'),
            yy.flatten(order='F'),
            zz.flatten(order='F')
        ], axis=1)

        # Set up surface areas (random positive values)
        self.surface_grid = np.abs(np.random.rand(self.nr, 3)) + 0.1

        # Set up permittivity (random positive values)
        self.eps = np.abs(np.random.rand(self.nr)) + 1.0

        # Set up potentials
        self.phi = np.random.rand(self.nr)
        self.phi_old = np.random.rand(self.nr)

        # Set up charges (can be positive or negative)
        self.free_charge = (np.random.rand(self.nr) - 0.5) * 1e-10
        self.fixed_charge = (np.random.rand(self.nr) - 0.5) * 1e-10

        # Set up Dirichlet potential for boundary
        self.lead_gate_potential = np.random.rand(self.nr)

        # Temperature
        self.kBT = 0.0259  # ~300K in eV

        # Average mode
        self.average_mode = 'arithmetic'

    def _old_implementation(self, J, B):
        """
        Reproduce the old implementation from poisson_init.py (commented out code).
        This is a direct port of the loop-based implementation.
        """
        def average_eps(eps1, eps2, mode='arithmetic'):
            if mode == 'arithmetic':
                return 0.5 * (eps1 + eps2)
            elif mode == 'harmonic':
                return 2.0 * eps1 * eps2 / (eps1 + eps2)
            elif mode == 'geometric':
                return np.sqrt(eps1 * eps2)

        average_mode = self.average_mode
        Nx = self.nx
        Ny = self.ny

        for gp_index in range(self.nr):
            if self.typ[gp_index] == "in":
                # x-direction fluxes
                flux_xm_J = self.surface_grid[gp_index, 0] * eps0 * \
                    average_eps(self.eps[gp_index - 1], self.eps[gp_index], mode=average_mode) / \
                    abs(self.grid_coord[gp_index, 0] - self.grid_coord[gp_index - 1, 0])
                flux_xm_B = flux_xm_J * (self.phi[gp_index - 1] - self.phi[gp_index])

                flux_xp_J = self.surface_grid[gp_index, 0] * eps0 * \
                    average_eps(self.eps[gp_index + 1], self.eps[gp_index], mode=average_mode) / \
                    abs(self.grid_coord[gp_index + 1, 0] - self.grid_coord[gp_index, 0])
                flux_xp_B = flux_xp_J * (self.phi[gp_index + 1] - self.phi[gp_index])

                # y-direction fluxes
                flux_ym_J = self.surface_grid[gp_index, 1] * eps0 * \
                    average_eps(self.eps[gp_index - Nx], self.eps[gp_index], mode=average_mode) / \
                    abs(self.grid_coord[gp_index - Nx, 1] - self.grid_coord[gp_index, 1])
                flux_ym_B = flux_ym_J * (self.phi[gp_index - Nx] - self.phi[gp_index])

                flux_yp_J = self.surface_grid[gp_index, 1] * eps0 * \
                    average_eps(self.eps[gp_index + Nx], self.eps[gp_index], mode=average_mode) / \
                    abs(self.grid_coord[gp_index + Nx, 1] - self.grid_coord[gp_index, 1])
                flux_yp_B = flux_yp_J * (self.phi[gp_index + Nx] - self.phi[gp_index])

                # z-direction fluxes
                flux_zm_J = self.surface_grid[gp_index, 2] * eps0 * \
                    average_eps(self.eps[gp_index - Nx * Ny], self.eps[gp_index], mode=average_mode) / \
                    abs(self.grid_coord[gp_index - Nx * Ny, 2] - self.grid_coord[gp_index, 2])
                flux_zm_B = flux_zm_J * (self.phi[gp_index - Nx * Ny] - self.phi[gp_index])

                flux_zp_J = self.surface_grid[gp_index, 2] * eps0 * \
                    average_eps(self.eps[gp_index + Nx * Ny], self.eps[gp_index], mode=average_mode) / \
                    abs(self.grid_coord[gp_index + Nx * Ny, 2] - self.grid_coord[gp_index, 2])
                flux_zp_B = flux_zp_J * (self.phi[gp_index + Nx * Ny] - self.phi[gp_index])

                # Jacobian diagonal element
                J[gp_index, gp_index] = -(flux_xm_J + flux_xp_J + flux_ym_J + flux_yp_J + flux_zm_J + flux_zp_J) \
                    + elementary_charge * self.free_charge[gp_index] * (-np.sign(self.free_charge[gp_index])) / self.kBT * \
                    np.exp(-np.sign(self.free_charge[gp_index]) * (self.phi[gp_index] - self.phi_old[gp_index]) / self.kBT)

                # Jacobian off-diagonal elements
                J[gp_index, gp_index - 1] = flux_xm_J
                J[gp_index, gp_index + 1] = flux_xp_J
                J[gp_index, gp_index - Nx] = flux_ym_J
                J[gp_index, gp_index + Nx] = flux_yp_J
                J[gp_index, gp_index - Nx * Ny] = flux_zm_J
                J[gp_index, gp_index + Nx * Ny] = flux_zp_J

                # B vector
                B[gp_index] = (flux_xm_B + flux_xp_B + flux_ym_B + flux_yp_B + flux_zm_B + flux_zp_B)
                B[gp_index] += elementary_charge * self.free_charge[gp_index] * \
                    np.exp(-np.sign(self.free_charge[gp_index]) * (self.phi[gp_index] - self.phi_old[gp_index]) / self.kBT) \
                    + elementary_charge * self.fixed_charge[gp_index]

            else:  # boundary points
                J[gp_index, gp_index] = 1.0

                if self.typ[gp_index] == "xmin":
                    J[gp_index, gp_index + 1] = -1.0
                    B[gp_index] = (self.phi[gp_index] - self.phi[gp_index + 1])
                elif self.typ[gp_index] == "xmax":
                    J[gp_index, gp_index - 1] = -1.0
                    B[gp_index] = (self.phi[gp_index] - self.phi[gp_index - 1])
                elif self.typ[gp_index] == "ymin":
                    J[gp_index, gp_index + Nx] = -1.0
                    B[gp_index] = (self.phi[gp_index] - self.phi[gp_index + Nx])
                elif self.typ[gp_index] == "ymax":
                    J[gp_index, gp_index - Nx] = -1.0
                    B[gp_index] = (self.phi[gp_index] - self.phi[gp_index - Nx])
                elif self.typ[gp_index] == "zmin":
                    J[gp_index, gp_index + Nx * Ny] = -1.0
                    B[gp_index] = (self.phi[gp_index] - self.phi[gp_index + Nx * Ny])
                elif self.typ[gp_index] == "zmax":
                    J[gp_index, gp_index - Nx * Ny] = -1.0
                    B[gp_index] = (self.phi[gp_index] - self.phi[gp_index - Nx * Ny])
                elif self.typ[gp_index] == "Dirichlet":
                    B[gp_index] = (self.phi[gp_index] - self.lead_gate_potential[gp_index])

            # Sign flip for NR iteration
            if B[gp_index] != 0:
                B[gp_index] = -B[gp_index]

    def test_nr_construct_vs_old_implementation(self):
        """Test that nr_construct produces the same results as the old implementation."""
        # Create matrices for old implementation
        J_old = lil_matrix((self.nr, self.nr), dtype=float)
        B_old = np.zeros(self.nr, dtype=float)

        # Run old implementation
        t = time.time()
        self._old_implementation(J_old, B_old)
        print(f'Old implementation took {time.time() - t:.4f} seconds')

        # Create matrices for new implementation
        J_new = lil_matrix((self.nr, self.nr), dtype=float)
        B_new = np.zeros(self.nr, dtype=float)

        # Select average function based on mode
        feps = {'harmonic': lambda eps1, eps2: 2.0 * eps1 * eps2 / (eps1 + eps2),
                'arithmetic': lambda eps1, eps2: 0.5 * (eps1 + eps2),
                'geometric': lambda eps1, eps2: np.sqrt(eps1 * eps2)}[self.average_mode]

        # Run new implementation
        t = time.time()
        nr_construct(
            jinout=J_new,
            binout=B_new,
            grid_dim=(self.nx, self.ny, self.nz),
            gridpoint_coords=self.grid_coord,
            gridpoint_typ=self.typ,
            gridpoint_surfarea=self.surface_grid,
            eps=self.eps,
            phi=self.phi,
            phi_=self.phi_old,
            free_chr=self.free_charge,
            fixed_chr=self.fixed_charge,
            dirichlet_pot=self.lead_gate_potential,
            eps0=eps0,
            beta=1.0 / self.kBT,
            feps=feps
        )
        print(f'New implementation took {time.time() - t:.4f} seconds')

        # Compare Jacobian matrices
        J_old_dense = J_old.toarray()
        J_new_dense = J_new.toarray()

        # Check diagonal elements
        diag_diff = np.abs(np.diag(J_old_dense) - np.diag(J_new_dense))
        max_diag_diff = np.max(diag_diff)
        print(f'Max diagonal difference: {max_diag_diff}')

        # Check off-diagonal elements
        off_diag_diff = np.abs(J_old_dense - J_new_dense)
        np.fill_diagonal(off_diag_diff, 0)
        max_off_diag_diff = np.max(off_diag_diff)
        print(f'Max off-diagonal difference: {max_off_diag_diff}')

        # Check B vectors
        B_diff = np.abs(B_old - B_new)
        max_B_diff = np.max(B_diff)
        print(f'Max B vector difference: {max_B_diff}')

        # Assert with reasonable tolerance
        rtol = 1e-10
        atol = 1e-15

        self.assertTrue(
            np.allclose(J_old_dense, J_new_dense, rtol=rtol, atol=atol),
            f'Jacobian matrices differ: max diagonal diff={max_diag_diff}, '
            f'max off-diagonal diff={max_off_diag_diff}'
        )
        self.assertTrue(
            np.allclose(B_old, B_new, rtol=rtol, atol=atol),
            f'B vectors differ: max diff={max_B_diff}'
        )

    def test_nr_construct_harmonic_average(self):
        """Test with harmonic average mode."""
        self.average_mode = 'harmonic'
        self.test_nr_construct_vs_old_implementation()

    def test_nr_construct_geometric_average(self):
        """Test with geometric average mode."""
        self.average_mode = 'geometric'
        self.test_nr_construct_vs_old_implementation()


if __name__ == '__main__':
    unittest.main()
