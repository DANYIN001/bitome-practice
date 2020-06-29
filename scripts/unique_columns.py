# built-in modules
from collections import Counter
from itertools import combinations

# third-party modules
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

# local modules
from bitome.core import Bitome

plt.rcParams.update({'font.size': 18})

test_bitome = Bitome.init_from_file('matrix_data/bitome.pkl')
bitome_matrix = test_bitome.extract(category='core sequence')
all_column_strs = np.array([
    '_'.join(map(str, bitome_matrix[:, i].nonzero()[0]))
    for i in tqdm(range(bitome_matrix.shape[1]))
])
unique_columns = list(set(all_column_strs))
unique_column_code_lookup = {col_str: i for i, col_str in enumerate(unique_columns)}
reverse_column_code = {v: k for k, v in unique_column_code_lookup.items()}
unique_column_locs = {}
for column_index, column_str in tqdm(enumerate(all_column_strs)):
    column_code = unique_column_code_lookup[column_str]
    current_locs = unique_column_locs.get(column_code)
    if current_locs is None:
        unique_column_locs[column_code] = [column_index]
    else:
        unique_column_locs[column_code] = current_locs + [column_index]
unique_column_counts = {col_code: len(locs) for col_code, locs in tqdm(unique_column_locs.items())}
counts_dict = Counter(list(unique_column_counts.values()))

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

threshold = 30000
sub_threshold = {k: v for k, v in unique_column_counts.items() if v < 30000}
sns.distplot(list(unique_column_counts.values()), kde=False, hist_kws={'log': True}, ax=ax1)
sns.distplot(list(sub_threshold.values()), kde=False, hist_kws={'log': True}, ax=ax2)
sns.despine()

ax1.set_title('All counts', loc='right')
ax2.set_title(f'Counts <{threshold}', loc='right')
ax.set_xlabel('Count of Unique Column', labelpad=30)
ax.set_ylabel('Count of Count', labelpad=40)
ax.set_xticks([])
ax.set_yticks([])
plt.show()

unique_column_tuples = [tuple(map(int, col.split('_'))) for col in unique_columns]
unique_column_lengths = list(map(len, unique_column_tuples))
median_unique_col_len = np.median(unique_column_lengths)
small_len = np.array(list(unique_column_counts.values()))[np.where(unique_column_lengths <= median_unique_col_len)[0]]
large_len = np.array(list(unique_column_counts.values()))[np.where(unique_column_lengths > median_unique_col_len)[0]]
small_len = [x for x in small_len if x < threshold]
large_len = [x for x in large_len if x < threshold]

fig, ax = plt.subplots()
sns.distplot(small_len, ax=ax, color=(0, 0, 1, 0.4), kde_kws={'linewidth': 3}, hist_kws={'log': True})
sns.distplot(large_len, ax=ax, color=(1, 0, 0, 0.4), kde_kws={'linewidth': 3}, hist_kws={'log': True})
sns.despine()
ax.set_xlabel('Count of Unique Column')
ax.set_ylabel('Count of Count (normalized)')
ax.yaxis.set_ticklabels([])
legend_elems = [
    Patch(facecolor=(1, 0, 0, 0.4), edgecolor=(1, 0, 0, 0.4), label='large'),
    Patch(facecolor=(0, 0, 1, 0.4), edgecolor=(0, 0, 1, 0.4), label='small')
]
fig.legend(handles=legend_elems, prop={'size': 16}, loc='upper right')
plt.xlim(left=0)
plt.show()

fig, ax = plt.subplots()
sns.distplot(unique_column_lengths, kde=False)
sns.despine()
ax.set_xlabel('Column Sum')
ax.set_ylabel('Count')
ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
plt.show()

# --- see which row indices contribute to the most unique columns ---
max_unique_col_sum = max(unique_column_lengths)
padded_column_vectors = [
    np.pad(
        np.array(tup), (0, max_unique_col_sum-len(tup)), mode='constant', constant_values=-1
    ).reshape(max_unique_col_sum, 1)
    for tup in unique_column_tuples
]
unique_column_matrix = np.hstack(padded_column_vectors)

new_unique_lengths = []
loc_maps = {}
for i in range(bitome_matrix.shape[0]):
    locs = np.where(unique_column_matrix == i)[1]
    loc_maps[i] = locs
for row_ind1, row_ind2 in tqdm(list(combinations(range(bitome_matrix.shape[0]), 2))):
    tup_inds_to_mod = list(set(loc_maps[row_ind1]).union(set(loc_maps[row_ind2])))
    removed_tups = list(np.delete(np.array(unique_column_tuples), tup_inds_to_mod))
    for tup_ind in tup_inds_to_mod:
        new_tup = set(unique_column_tuples[tup_ind])
        new_tup.discard(row_ind1)
        new_tup.discard(row_ind2)
        removed_tups.append(tuple(new_tup))
    new_unique_lengths.append(len(set(removed_tups)))

core_inds = [i for i, cat in enumerate(test_bitome.matrix_row_categories) if cat == 'core sequence']
core_row_labels = np.array(test_bitome.matrix_row_labels)[np.array(core_inds)]

