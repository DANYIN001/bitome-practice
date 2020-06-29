# --- MEANT TO BE RUN ON NERSC ---

# built-in modules
from itertools import combinations
from timeit import default_timer as timer

# third-party modules
import h5py
import numpy as np
from scipy.sparse import load_npz
from scipy.spatial.distance import hamming

# note that the matrix in the file is transposed so that the feature "rows" are actually columns
matrix_load_start = timer()
bitome_matrix = load_npz('matrix_data/bitome_matrix_1637.npz').T
bitome_matrix = bitome_matrix[:100, :10000]
n_rows = bitome_matrix.shape[0]
matrix_load_end = timer()
print(f'Matrix loaded in {matrix_load_end-matrix_load_start} seconds')

# generate all of the possible combinations of non-equivalent indices (one triangular half of distance matrix)
dist_start = timer()
row_index_pairs = combinations(list(range(n_rows)), 2)
dist_matrix = np.zeros(shape=(n_rows, n_rows))
for row_index_1, row_index_2 in row_index_pairs:
    dist = hamming(bitome_matrix[row_index_1, :].todense(), bitome_matrix[row_index_2, :].todense())
    dist_matrix[row_index_1, row_index_2] = dist
    dist_matrix[row_index_2, row_index_1] = dist
dist_end = timer()
print(f'Distance matrix generated in {dist_end-dist_start} seconds')

data_save_start = timer()
h5f = h5py.File('dist.h5', 'w')
h5f.create_dataset('dist_matrix', data=dist_matrix, compression='gzip', chunks=True)
h5f.close()
data_save_end = timer()
print(f'Distance matrix saved in h5 format in {data_save_end-data_save_start} seconds')

print('Complete!')
