'''
benchmark the speed-up of function `indexing_atom_on_grid` in `dpnegf.negf.poisson_init.Grid`
'''
import unittest
import time

import numpy as np

def indexing_atom_on_grid(grids, xa, ya, za, thr=1e-6):
    xa, ya, za = map(np.asarray, (xa, ya, za))
    assert xa.shape == ya.shape == za.shape, "xa, ya, za must have the same shape"
    na, = xa.shape
    # allocate the indexing dictionary
    indexing = dict(zip(range(na), np.zeros(na, dtype=int)))
    for ia, (x, y, z) in enumerate(zip(xa, ya, za)):
        # find the index of the atom in the grid
        index = np.where(
            (np.abs(grids[:, 0] - x) < thr) &
            (np.abs(grids[:, 1] - y) < thr) &
            (np.abs(grids[:, 2] - z) < thr)
        )[0]
        if len(index) > 0:
            indexing[ia] = index[0]
    return indexing

def get_atom_index(grids, xa, ya, za):
    swap = {}
    Na = len(xa)
    Np = len(grids)
    for atom_index in range(Na):
        for gp_index in range(Np):
            if abs(xa[atom_index]-grids[gp_index][0])<1e-3 and \
               abs(ya[atom_index]-grids[gp_index][1])<1e-3 and \
               abs(za[atom_index]-grids[gp_index][2])<1e-3:
               swap.update({atom_index:gp_index})
    return swap

class TestGrid(unittest.TestCase):
    def setUp(self):
        self.grid_coord = np.random.rand(10000, 3)  # 10,000 grid points
    
    def test(self):
        xa = self.grid_coord[:, 0]
        ya = self.grid_coord[:, 1]
        za = self.grid_coord[:, 2]
        
        start_time = time.time()
        ind1 = indexing_atom_on_grid(self.grid_coord, xa, ya, za)
        end_time = time.time()
        print(f"Time taken for indexing_atom_on_grid: {end_time - start_time} seconds")
        
        start_time = time.time()
        ind2 = get_atom_index(self.grid_coord, xa, ya, za)
        end_time = time.time()
        print(f"Time taken for get_atom_index: {end_time - start_time} seconds")      
        
        self.assertDictEqual(ind1, ind2, "The indexing results should be the same for both methods.")
        
if __name__ == '__main__':
    unittest.main()