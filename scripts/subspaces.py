# third-party modules
import h5py
import matplotlib.pyplot as plt
import numpy as np

# load the HDF5 group file
svd_matrices_file = h5py.File('matrix_data/3_svd_1668/svd_data.h5')

# create singular value spectrum
s = svd_matrices_file['S'][:]
rank = len(np.where(s != 0.0))
singular_values = s.toarray().flatten()
# svds orders S from low to high
singular_values_nz = np.flip(singular_values[:rank])
pct_var_exp = np.cumsum(singular_values_nz)/singular_values_nz.sum()
x_range = list(range(len(pct_var_exp)))

plt.plot(x_range, pct_var_exp)
plt.xlabel('Singular Vector')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by Singular Modes of B')
plt.hlines(0.95, 0, 960, colors='r')
plt.vlines(960, 0, 0.95, colors='r')
plt.show()
