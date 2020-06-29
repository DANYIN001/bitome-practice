# built-in modules
from collections import deque

# third-party modules
from Bio.Alphabet import IUPAC
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0 as zeroth_order_bessel
from scipy.stats import pearsonr

# local modules
from bitome.core import Bitome


def von_mises(data):
    max_val = max(data)
    normalized = (data * 2 * np.pi / max_val) - np.pi
    bins, kde = von_mises_fft_kde(data=normalized, kappa=10, n_bins=100)
    final_bins = (bins + np.pi) * max_val / (2 * np.pi) / 1000
    return final_bins, kde


# based on https://stackoverflow.com/questions/28839246/scipy-gaussian-kde-and-circular-data
def von_mises_pdf(x, mu, kappa):
    return np.exp(kappa * np.cos(x - mu)) / (2. * np.pi * zeroth_order_bessel(kappa))


def von_mises_fft_kde(data, kappa, n_bins):
    bins = np.linspace(-np.pi, np.pi, n_bins + 1, endpoint=True)
    hist_n, bin_edges = np.histogram(data, bins=bins)
    bin_centers = np.mean([bin_edges[1:], bin_edges[:-1]], axis=0)
    kernel = von_mises_pdf(bin_centers, 0, kappa)
    kde = np.fft.fftshift(np.fft.irfft(np.fft.rfft(kernel) * np.fft.rfft(hist_n)))
    kde /= np.trapz(kde, x=bin_centers)
    return bin_centers, kde


test_bitome = Bitome.init_from_file('matrix_data/bitome.pkl')

nt_byte_sizes = test_bitome.nt_byte_sizes
nt_byte_sizes_flat = np.asarray(nt_byte_sizes).flatten()
window_size = 200000
seq_len = len(nt_byte_sizes_flat)
window_sums = np.array([np.sum(nt_byte_sizes_flat[i:i+window_size]) for i in range(0, seq_len, window_size)][:-1])
mean_window_sum = np.mean(window_sums)
window_sums_norm = (window_sums-np.mean(window_sums))/np.std(window_sums)

cds_genes = [gene for gene in test_bitome.genes if gene.gene_type == 'CDS']
cai_by_window = []
for low_bound in range(0, seq_len, window_size):
    window_genes = []
    for cds_gene in cds_genes:
        cds_gene_location = cds_gene.location
        if cds_gene_location.start.position > low_bound and cds_gene_location.end.position < low_bound + window_size:
            window_genes.append(cds_gene)
    window_cais = [window_gene.cai for window_gene in window_genes]
    window_cai_avg = np.mean(window_cais)
    cai_by_window.append(window_cai_avg)
cai_by_window = np.array(cai_by_window[:-1])
cai_by_window_norm = (cai_by_window-np.mean(cai_by_window))/np.std(cai_by_window)

corr, pval = pearsonr(cai_by_window, window_sums)


def re_order(values):
    ori = test_bitome.origin.location.start.position
    middle_bucket = np.floor(len(values)/2)
    ori_bucket = np.floor((ori/len(test_bitome.sequence))*len(values))
    shift = middle_bucket - ori_bucket
    vals = deque(values)
    vals.rotate(int(shift))
    return np.array(vals)


x_range = range(0, int(seq_len/1000), int(window_size/1000))[:-1]
x_range_ori = range(int(-12*window_size/1000), int(12*window_size/1000), int(window_size/1000))[:-1]
plt.figure(figsize=(8, 5))
plt.plot(x_range_ori, re_order(window_sums_norm), label='Information')
plt.plot(x_range_ori, re_order(cai_by_window_norm), label='CAI')
plt.title('Information Content and Codon Adaptation across E. Coli K-12 MG1655 Genome')
plt.legend(loc='lower right')
plt.xlabel('Genome position from ORI (kb)')
plt.ylabel('Z score')
plt.text(1450, 2, f'Pearson R: {corr:.2f}\np-value: {pval:.4f}')
plt.show()

amino_acid_letters = IUPAC.protein.letters
sec_struct_row_names = [f'secondary_structure_{lett}' for lett in ['B', 'E', 'I', 'G', 'H', 'T', 'S', '-']]
amino_acid_row_names = [f'amino_acid_{lett}' for lett in amino_acid_letters]
cog_letters = [
    'E', 'M', 'A', 'J', 'H', 'C', 'L', 'V', 'G', 'S', 'U',
    'R', 'W', 'K', 'X', 'N', 'Q', 'I', 'D', 'P', 'T', 'O', 'F'
]
cog_names = [f'COG_{lett}' for lett in cog_letters]
mrna_struct_rows = ['mRNA_structure']
all_row_names = sec_struct_row_names + amino_acid_row_names + mrna_struct_rows + cog_names
corr_dict = {}
for name in all_row_names:
    sub_mat = test_bitome.extract(row_labels=[name], base_name=True)
    name_window_sums = np.array([np.sum(sub_mat[:, i:i+window_size]) for i in range(0, seq_len, window_size)][:-1])
    name_window_sums_norm = (name_window_sums - np.mean(name_window_sums)) / np.std(name_window_sums)
    corr, pval = pearsonr(cai_by_window, name_window_sums)
    corr_dict[name] = {
        'corr': corr,
        'pval': pval,
        'window_sums': name_window_sums_norm
    }

name_to_plot = 'mRNA_structure'
plt.plot(x_range, corr_dict[name_to_plot]['window_sums'], label=name_to_plot)
plt.plot(x_range, cai_by_window_norm, label='CAI')
plt.title(f'{name_to_plot} and CAI across E. Coli K-12 MG1655 Genome')
plt.legend(loc='lower right')
plt.xlabel('Genome position (kb)')
plt.ylabel('Z score')
plt.text(3500, -1.75, f"Pearson R: {corr_dict[name_to_plot]['corr']:.2f}\np-value: {corr_dict[name_to_plot]['pval']:.2E}")
plt.show()

amino_acid_dict = {name: dict_val for name, dict_val in corr_dict.items() if 'amino_acid' in name}
name_corr_tups = [(name, dict_val['corr']) for name, dict_val in amino_acid_dict.items()]
sorted_amino_acid_corrs = sorted(name_corr_tups, key=lambda tup: tup[1], reverse=True)

aa_sum_tups = [
    (lett, test_bitome.extract(row_labels=[f'amino_acid_{lett}'], base_name=True).sum())
    for lett in amino_acid_letters
]

aa_corr_aas = [tup[0][-1] for tup in sorted_amino_acid_corrs]
aa_corrs = [tup[1] for tup in sorted_amino_acid_corrs]

aa_sums_in_corr_order = []
for aa in aa_corr_aas:
    aa_sum_tup = [tup for tup in aa_sum_tups if tup[0] == aa][0]
    aa_sums_in_corr_order.append(aa_sum_tup[1])
full_aa_sum = sum(aa_sums_in_corr_order)
aa_sums_norm = np.array(aa_sums_in_corr_order)/full_aa_sum

# excludes the last two that have neg corrs barely
aa_corr_aas = aa_corr_aas[:-2]
aa_corrs = aa_corrs[:-2]
aa_sums_norm = aa_sums_norm[:-2]

fig, ax1 = plt.subplots()
bar_x = np.arange(len(aa_corr_aas)) * 2
ax1.bar(bar_x, aa_corrs, tick_label=aa_corr_aas, width=-0.7, color='blue', align='edge')
ax1.set_xlabel('Amino Acid')
ax1.set_ylabel('Pearson R with CAI', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Relative Frequency', color='olivedrab')
ax2.bar(bar_x+0.75, aa_sums_norm, width=-0.7, color='olivedrab', align='edge')
ax2.tick_params(axis='y', labelcolor='olivedrab')

plt.title('Amino Acid/CAI Correlations and Amino Acid Relative\n Frequencies for E. Coli K-12 MG1655')
plt.tight_layout()
plt.show()

sec_struct_dict = {name: dict_val for name, dict_val in corr_dict.items() if 'secondary_structure' in name}
ss_name_corr_tups = [(name, dict_val['corr']) for name, dict_val in sec_struct_dict.items()]
sorted_sec_struct_corrs = sorted(ss_name_corr_tups, key=lambda tup: tup[1], reverse=True)

ss_sum_tups = [
    (lett, test_bitome.extract(row_labels=[f'secondary_structure_{lett}'], base_name=True).sum())
    for lett in ['B', 'E', 'I', 'G', 'H', 'T', 'S', '-']
]

ss_corr_ss = [tup[0][-1] for tup in sorted_sec_struct_corrs]
ss_corrs = [tup[1] for tup in sorted_sec_struct_corrs]

ss_sums_in_corr_order = []
for ss in ss_corr_ss:
    ss_sum_tup = [tup for tup in ss_sum_tups if tup[0] == ss][0]
    ss_sums_in_corr_order.append(ss_sum_tup[1])
full_ss_sum = sum(ss_sums_in_corr_order)
ss_sums_norm = np.array(ss_sums_in_corr_order)/full_ss_sum

# excludes the last one that have neg corrs barely
ss_corr_ss = ss_corr_ss[:-1]
ss_corrs = ss_corrs[:-1]
ss_sums_norm = ss_sums_norm[:-1]

fig, ax3 = plt.subplots()
bar_x = np.arange(len(ss_corr_ss)) * 2
ax3.set_xlabel('Secondary Structure (DSSP symbols)')
ax3.bar(bar_x, ss_corrs, tick_label=ss_corr_ss, width=-0.7, color='blue', align='edge')
ax3.set_ylabel('Pearson R with CAI', color='blue')
ax3.tick_params(axis='y', labelcolor='blue')

ax4 = ax3.twinx()
ax4.set_ylabel('Relative Frequency', color='olivedrab')
ax4.bar(bar_x+0.75, ss_sums_norm, width=-0.7, color='olivedrab', align='edge')
ax4.tick_params(axis='y', labelcolor='olivedrab')

plt.title('Secondary Structure/CAI Correlations and Amino Acid Relative\n Frequencies for E. Coli K-12 MG1655')
plt.tight_layout()
plt.show()

cog_dict = {name: dict_val for name, dict_val in corr_dict.items() if 'COG' in name}
cog_name_corr_tups = [(name, dict_val['corr']) for name, dict_val in cog_dict.items()]
sorted_cog_corrs = sorted(cog_name_corr_tups, key=lambda tup: tup[1], reverse=True)

cog_sum_tups = [
    (lett, test_bitome.extract(row_labels=[f'COG_{lett}'], base_name=True).sum())
    for lett in cog_letters
]

cog_cor_cog = [tup[0][-1] for tup in sorted_cog_corrs]
cog_corrs = [tup[1] for tup in sorted_cog_corrs]

cog_sums_in_corr_order = []
for cog in cog_cor_cog:
    cog_sum_tup = [tup for tup in cog_sum_tups if tup[0] == cog][0]
    cog_sums_in_corr_order.append(cog_sum_tup[1])
full_cog_sum = sum(cog_sums_in_corr_order)
cog_sum_norm = np.array(cog_sums_in_corr_order)/full_cog_sum

# excludes the last one that have neg corrs barely
cog_cor_cog = cog_cor_cog[:-1]
cog_corrs = cog_corrs[:-1]
cog_sum_norm = cog_sum_norm[:-1]

fig, ax5 = plt.subplots()
bar_x = np.arange(len(cog_cor_cog)) * 2
ax5.set_xlabel('COG')
ax5.bar(bar_x, cog_corrs, tick_label=cog_cor_cog, width=-0.7, color='blue', align='edge')
ax5.set_ylabel('Pearson R with CAI', color='blue')
ax5.tick_params(axis='y', labelcolor='blue')

ax6 = ax5.twinx()
ax6.set_ylabel('Relative Frequency', color='olivedrab')
ax6.bar(bar_x+0.75, cog_sum_norm, width=-0.7, color='olivedrab', align='edge')
ax6.tick_params(axis='y', labelcolor='olivedrab')

plt.title('COG/CAI Correlations and Amino Acid Relative\n Frequencies for E. Coli K-12 MG1655')
plt.tight_layout()
plt.show()


