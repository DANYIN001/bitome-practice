from timeit import default_timer as timer

import h5py
from numpy import diag, matmul, round
from scipy.sparse import load_npz
from scipy.sparse.linalg import svds

# load the current bitome matrix; assumes it is stored as a csr_matrix in .npz format
matrix_load_start = timer()
bitome_matrix = load_npz('../matrix_data/bitome_matrix_1668.npz')
bitome_matrix = bitome_matrix[:1700, :]
matrix_load_end = timer()
print(f'Matrix loaded in {matrix_load_end-matrix_load_start} seconds')

# run the matrix decomposition via SVD from scipy.sparse
svd_start = timer()
n_cols = bitome_matrix.shape[1]
u, s, vt = svds(bitome_matrix, k=n_cols-1)
svd_end = timer()
print(f'SVD calculated in {svd_end-svd_start} seconds')

# save the arrays as hd5 archives; don't close the file, we will save more later
data_save_start = timer()
h5f = h5py.File('svd_data.h5', 'w')
h5f.create_dataset('U', data=u, compression='gzip', chunks=True)
h5f.create_dataset('S', data=s, compression='gzip', chunks=True)
h5f.create_dataset('Vt', data=vt, compression='gzip', chunks=True)
data_save_end = timer()
print(f'U, S, and Vt matrices saved in hd5 format in {data_save_end-data_save_start} seconds')

reconstruct_start = timer()
s_diag = diag(s)
bitome_reconstructed = round(matmul(matmul(u, s_diag), vt))
h5f.create_dataset('bitome_reconstructed', data=bitome_reconstructed, compression='gzip', chunks=True)
h5f.close()
reconstruct_end = timer()
print(f'Matrix reconstructed and saved in {reconstruct_end - reconstruct_start} seconds')

diff_start = timer()
diff = bitome_matrix - bitome_reconstructed
if abs(diff.sum()) > 0:
    print('Not all elements match between bitome matrix and reconstructed product matrix')
diff_end = timer()
print(f'Reconstruction compared in {diff_end - diff_start} seconds')

print(f'All tasks complete in {diff_end-matrix_load_start} seconds')


