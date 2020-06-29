# built-in modules
from pathlib import Path

# third-party modules
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

# local modules
from bitome.core import Bitome

DATA_DIR = Path('matrix_data')
svd_matrices_file = h5py.File(Path(DATA_DIR, 'dist_full.h5'))
dist_matrix = svd_matrices_file['dist_matrix'][:]
corr_matrix = 1 - dist_matrix

test_bitome = Bitome.init_from_file('matrix_data/bitome.pkl')
matrix_labels = test_bitome.matrix_labels

# want a histogram that excludes the symmetric triangle of distance matrix
unique_dists = []
for i, row in enumerate(dist_matrix):
    unique_dists += list(row[i+1:])

sns.distplot(unique_dists, kde=False, hist_kws={'log': True})
plt.title('Distribution of Hamming Distances between Bitome Rows')
plt.xlabel('Hamming Distance')
plt.ylabel('Count')
plt.show()

sns.heatmap(corr_matrix)
plt.title('Bitome Correlation Matrix (1 - Hamming distance)')
plt.show()
