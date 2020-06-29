# built-in modules
import os
from pathlib import Path
import pickle
from typing import Tuple

# third-party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, hstack, save_npz
from tqdm import tqdm

# local modules
from bitome.core import Bitome

test_bitome = Bitome.init_from_file('matrix_data/bitome.pkl')
TEMP_PATH = Path('matrix_data', 'temp.npz')
REDO_CALCULATIONS = False

binary_matrix = test_bitome.matrix
pre_compressed_matrix = test_bitome.pre_compressed_matrix
genome_length = binary_matrix.shape[1]
origin_start = test_bitome.origin.location.start.position


def matrix_size(matrix: csc_matrix) -> Tuple[float, float]:
    """
    Given a matrix or sub-matrix, returns the compressed and non-compressed file sizes
    """
    save_npz(TEMP_PATH, matrix, compressed=False)
    non_compress_size = os.path.getsize(TEMP_PATH)
    os.remove(TEMP_PATH)

    save_npz(TEMP_PATH, matrix, compressed=True)
    compress_size = os.path.getsize(TEMP_PATH)
    os.remove(TEMP_PATH)

    return non_compress_size, compress_size


# --- get the compressed and non-compressed sizes of the binary and pre-compressed matrices ---
binary_size, binary_size_compressed = matrix_size(binary_matrix)
pre_compressed_size, pc_compressed_size = matrix_size(pre_compressed_matrix)
size_vector = np.array([binary_size, pre_compressed_size, binary_size_compressed, pc_compressed_size])
result_mat = np.zeros((4, 4))
for (i, j), _ in np.ndenumerate(result_mat):
    result_mat[i, j] = size_vector[i]/size_vector[j]
full_matrix_size_result_df = pd.DataFrame(
    result_mat,
    index=['binary', 'pre-comp', 'binary (comp)', 'pre-comp (comp)'],
    columns=['binary', 'pre-comp', 'binary (comp)', 'pre-comp (comp)']
)


# --- location-wise compression analysis ---

# perform sensitivity analysis to test a few different sliding window sizes for compression analysis
window_sizes = [10000, 25000, 60000, 100000, 200000, 400000, 600000]
step_size = 5000
if REDO_CALCULATIONS:

    # set up a results holding dict
    binary_result_dict = {}
    pre_compressed_result_dict = {}

    for window_size in window_sizes:
        print(f'Window size: {window_size}')

        left_end = int(origin_start - genome_length/2)
        right_end = int(origin_start + genome_length/2 + window_size - genome_length)

        binary_matrix_ori_center = hstack([binary_matrix[:, left_end:], binary_matrix[:, :right_end]])
        pre_compressed_matrix_ori_center = hstack(
            [pre_compressed_matrix[:, left_end:], pre_compressed_matrix[:, :right_end]]
        )

        binary_size_tuples = []
        pre_compressed_size_tuples = []
        for window_start in tqdm(range(0, binary_matrix_ori_center.shape[1], step_size)):
            binary_matrix_window = binary_matrix_ori_center[:, window_start:window_start+window_size]
            pre_compressed_matrix_window = pre_compressed_matrix_ori_center[:, window_start:window_start+window_size]
            binary_size_tuples.append(matrix_size(binary_matrix_window))
            pre_compressed_size_tuples.append(matrix_size(pre_compressed_matrix_window))

        binary_result_dict[window_size] = binary_size_tuples
        pre_compressed_result_dict[window_size] = pre_compressed_size_tuples

    with open('matrix_data/binary_compress_temp.pkl', 'wb') as handle:
        pickle.dump(binary_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('matrix_data/pre_compress_temp.pkl', 'wb') as handle:
        pickle.dump(pre_compressed_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('matrix_data/binary_compress_temp.pkl', 'rb') as handle:
    binary_result_dict = pickle.load(handle)

with open('matrix_data/pre_compress_temp.pkl', 'rb') as handle:
    pre_compressed_result_dict = pickle.load(handle)

# --- plotting the results ---

# make sure we perfectly cut off the overlap portion plus/minus the origin
# NOTE: had the right_end calculation in the window size loop wrong when last generating the result dicts;
# need to clip off a bunch at the right end (everything from true right end on)
x_range = range(-int(genome_length/2/1000), int(genome_length/2/1000), int(step_size/1000))
num_data_points = len(x_range)

for window_size in window_sizes:

    binary_results = binary_result_dict[window_size][:num_data_points]
    binary_sizes, binary_sizes_comp = zip(*binary_results)
    pre_compressed_results = pre_compressed_result_dict[window_size][:num_data_points]
    pc_sizes, pc_sizes_comp = zip(*pre_compressed_results)
    binary_to_pre_comp = np.array(binary_sizes)/np.array(pc_sizes)
    binary_to_binary_comp = np.array(binary_sizes)/np.array(binary_sizes_comp)
    binary_to_pc_comp = np.array(binary_sizes)/np.array(pc_sizes_comp)

    fig, axs = plt.subplots(3, 1, sharex=True)
    for ax, comp_ratios, title in zip(
                axs,
                [binary_to_pre_comp, binary_to_binary_comp, binary_to_pc_comp],
                ['Binary to Pre-Compressed', 'Binary to Compressed', 'Binary to Pre-Compressed (Compressed)']
            ):
        ax.plot(x_range, comp_ratios)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title, loc='right')
    axs[2].set_xlabel('Genome position (kbp from origin)')
    axs[1].set_ylabel('Compression ratio')
    plt.tight_layout()
    plt.show()
