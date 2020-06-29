# built-in modules
from pathlib import Path

# third-party modules
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns

# local modules
from bitome.core import Bitome

lambda_bitome = Bitome(Path('data', 'NC_001416.1.gb'))
lambda_bitome.load_data(regulon_db=False)
lambda_bitome.load_matrix()

cov_bitome = Bitome(Path('data', 'NC_045512.2.gb'))
cov_bitome.load_data(regulon_db=False)
cov_bitome.load_matrix()

# --- Column Sums ---
fig, axs = plt.subplots(2, 1, sharex='all')
ax1, ax2 = axs
sns.distplot(
    np.asarray(lambda_bitome.matrix.sum(axis=0)).flatten(),
    bins=np.arange(0, 15),
    kde=False,
    hist_kws={
        'log': True,
        'rwidth': 0.9
    },
    ax=ax1
)
sns.distplot(
    np.asarray(cov_bitome.matrix.sum(axis=0)).flatten(),
    bins=np.arange(0, 15),
    kde=False,
    hist_kws={
        'log': True,
        'rwidth': 0.9
    },
    ax=ax2
)
plt.xlabel('Information (bits)', fontsize=24)
plt.xticks([2, 4, 6, 8, 10, 12, 14, 16])
ax1.set_title('Lambda', fontsize=22)
ax2.set_title('Coronavirus', fontsize=22)
ax1.tick_params(axis='both', labelsize='20')
ax2.tick_params(axis='both', labelsize='20')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.show()

# -- Feature Coverages ---
features_to_extract = {
    'base': 'nucleotide',
    'gene_CDS': 'coding gene',
    'amino_acid_L': 'leucine'
}
lambda_seq_length = len(lambda_bitome.sequence)
cov_seq_length = len(cov_bitome.sequence)
lambda_coverages = []
cov_coverages = []
for feat_name in features_to_extract.keys():
    lambda_feat_mat = lambda_bitome.extract(row_labels=[feat_name], base_name=True)
    cov_feat_mat = cov_bitome.extract(row_labels=[feat_name], base_name=True)
    lambda_feat_vector = np.asarray(lambda_feat_mat.sum(axis=0)).flatten()
    cov_feat_vector = np.asarray(cov_feat_mat.sum(axis=0)).flatten()
    lambda_coverages.append(len(np.where(lambda_feat_vector > 0)[0]))
    cov_coverages.append(len(np.where(cov_feat_vector > 0)[0]))

lambda_coverage_pcts = (np.array(lambda_coverages)/lambda_seq_length)*100
cov_coverage_pcts = (np.array(cov_coverages)/cov_seq_length)*100

lambda_names_sorted, lambda_coverages_sorted = zip(*sorted(
    zip(features_to_extract, lambda_coverage_pcts),
    key=lambda tup: tup[1]
))
cov_names_sorted, cov_coverages_sorted = zip(*sorted(
    zip(features_to_extract, cov_coverage_pcts),
    key=lambda tup: tup[1]
))

fig, axs = plt.subplots(2, 1, sharex='all')
ax1, ax2 = axs

y_lambda = np.arange(len(lambda_coverages_sorted)) * 1.5
ax1.barh(y_lambda, lambda_coverages_sorted, log=True, tick_label=lambda_names_sorted)
ax1.set_xlim(right=150)
for y_val, pct in zip(y_lambda, lambda_coverages_sorted):
    if pct > 10:
        label = f'{pct:.0f} %'
    elif pct > 1:
        label = f'{pct:.1f} %'
    elif pct >= 0.1:
        label = f'{pct:.2f} %'
    else:
        label = f'{pct:.3f} %'
    ax1.text(pct*1.02, y_val, label, fontsize=14)

y_cov = np.arange(len(cov_coverages_sorted)) * 1.5
ax2.barh(y_cov, cov_coverages_sorted, log=True, tick_label=cov_names_sorted)
ax2.set_xlim(right=150)
for y_val, pct in zip(y_cov, cov_coverages_sorted):
    if pct > 10:
        label = f'{pct:.0f} %'
    elif pct > 1:
        label = f'{pct:.1f} %'
    elif pct >= 0.1:
        label = f'{pct:.2f} %'
    else:
        label = f'{pct:.3f} %'
    ax2.text(pct*1.02, y_val, label, fontsize=14)


def x_axis_formatter(x, _):
    if x < 1:
        tick_label = f'{x:.1f}'
    else:
        tick_label = f'{x:.0f}'
    return tick_label


ax1.set_title('Lambda', fontsize=22)
ax2.set_title('Coronavirus', fontsize=22)
ax1.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
ax2.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(axis='both', labelsize='20')
ax2.tick_params(axis='both', labelsize='20')
plt.xlabel('Sequence Coverage (%)', fontsize=24)
plt.show()
