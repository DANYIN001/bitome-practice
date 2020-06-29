# built-in modules
from itertools import product
from pathlib import Path
from typing import List, Tuple

# third-party modules
from Bio.Alphabet import IUPAC
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from tqdm import tqdm
from scipy.stats import iqr
import seaborn as sns

# local modules
from bitome.core import Bitome
from bitome.utilities import bits_per_bp_plot

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']

# load the Bitome object that serves as our "database" for all of the features, as well as the matrix
test_bitome = Bitome.init_from_file('matrix_data/bitome.pkl')

FIG_PATH = Path('figures', 'figure_3')

promoters = [promoter for promoter in test_bitome.promoters if promoter.transcription_unit is not None]

dists_to_minus_10 = []
dists_to_minus_35 = []
spacer_lengths = []
for promoter in promoters:
    # assume promoters have both boxes if one is present
    box_10_location = promoter.box_10_location
    box_35_location = promoter.box_35_location
    if box_10_location is not None:
        # define the average position of the -10/-35 boxes; need to subtract 1 from the end because it is NOT inclusive
        minus_10_center = abs(box_10_location.start.position + box_10_location.end.position-1)/2
        minus_35_center = abs(box_35_location.start.position + box_35_location.end.position-1)/2
        tss = promoter.tss_location.start.position
        dists_to_minus_10.append(abs(tss-minus_10_center))
        dists_to_minus_35.append(abs(tss-minus_35_center))
        
        # also want to get the distance between the edges of the things (hacky way where I don't have to check strand)
        spacer_length = min(np.abs([
            box_10_location.start.position - box_35_location.end.position,
            box_10_location.end.position - box_35_location.start.position
        ]))
        spacer_lengths.append(spacer_length)

dist_to_minus_10 = np.array(dists_to_minus_10) * -1
dist_to_minus_35 = np.array(dists_to_minus_35) * -1
dist_to_minus_10_median = np.median(dist_to_minus_10)
dist_to_minus_35_median = np.median(dist_to_minus_35)

# --- want to define the inter-TU vs TU regions ---
tu_sub_matrix = test_bitome.extract(row_labels=['TU'], base_name=True)
tu_vector = np.asarray(tu_sub_matrix.sum(axis=0)).flatten()
inter_tu_locs = np.where(tu_vector == 0)[0]

# want to get tuple ranges for the inter-TU regions
current_start = inter_tu_locs[0]
previous = current_start
remaining_locs = inter_tu_locs[1:]
inter_tu_ranges = []
while len(remaining_locs) > 0:
    if remaining_locs[0] - previous == 1:
        previous = remaining_locs[0]
        remaining_locs = remaining_locs[1:]
        continue
    else:
        inter_tu_ranges.append((current_start, previous))
        current_start = remaining_locs[0]
        previous = current_start
        remaining_locs = remaining_locs[1:]
inter_tu_ranges = [tup for tup in inter_tu_ranges if tup[0] < tup[1]]

# --- operon- and TU-based bits per bp calculations ---
operon_intergenic_ranges = []
for operon in test_bitome.operons:

    operon_tus = operon.transcription_units
    operon_genes = []
    for tu in operon_tus:
        operon_genes += tu.genes
    operon_genes = list(set(operon_genes))

    if not operon_genes:
        continue

    intergenic_ranges = []
    gene_ranges = [(gene.location.start.position, gene.location.end.position) for gene in operon_genes]

    strand = operon.location.strand
    if strand == 1:
        sorted_ranges_left = sorted(gene_ranges, key=lambda tup: tup[0])
        for i, current_range in enumerate(sorted_ranges_left):
            if i == 0:
                continue
            else:
                previous_range = sorted_ranges_left[i-1]
                if current_range[0] > previous_range[1]:
                    operon_intergenic_ranges.append((previous_range[1], current_range[0]))
    else:
        sorted_ranges_right = sorted(gene_ranges, key=lambda tup: tup[1], reverse=True)
        for i, current_range in enumerate(sorted_ranges_right):
            if i == 0:
                continue
            else:
                previous_range = sorted_ranges_right[i-1]
                if current_range[1] < previous_range[0]:
                    operon_intergenic_ranges.append((current_range[1], previous_range[0]))

five_prime_utr_ranges = []
three_prime_utr_ranges = []
tus = [tu for tu in test_bitome.transcription_units if tu.genes and tu.promoter is not None]
for tu in tus:

    tu_genes = tu.genes
    tu_strand = tu.location.strand
    if tu_strand == 1:
        translation_start = min([gene.location.start.position for gene in tu_genes])
        translation_end = max([gene.location.end.position for gene in tu_genes])
        five_prime_utr_range = tu.tss, translation_start
        three_prime_utr_range = translation_end, tu.tts
    else:
        translation_start = max([gene.location.end.position for gene in tu_genes])
        translation_end = min([gene.location.start.position for gene in tu_genes])
        five_prime_utr_range = translation_start, tu.tss
        three_prime_utr_range = tu.tts, translation_end

    # we may not actually have a five or three prime UTR
    if five_prime_utr_range[0] < five_prime_utr_range[1]:
        five_prime_utr_ranges.append(five_prime_utr_range)
    if three_prime_utr_range[0] < three_prime_utr_range[1]:
        three_prime_utr_ranges.append(three_prime_utr_range)

five_prime_utr_lengths = [utr_5_range[1] - utr_5_range[0] for utr_5_range in five_prime_utr_ranges]
three_prime_utr_lengths = [utr_3_range[1] - utr_3_range[0] for utr_3_range in three_prime_utr_ranges]


def outlier_indices(lengths: list) -> np.array:
    """
    Given a list of UTR lengths, return indices of outliers based on 1.5*IQR
    :param list lengths: the UTR lengths
    :return np.array outlier_indices: the indices of the outliers as defined above
    """
    length_iqr = iqr(lengths)
    q1 = np.percentile(lengths, 25)
    q3 = np.percentile(lengths, 75)
    indices = []
    for idx, length in enumerate(lengths):
        if length < (q1-length_iqr*1.5) or length > (q3+length_iqr*1.5):
            indices.append(idx)
    return indices


five_prime_utr_outliers = outlier_indices(five_prime_utr_lengths)
three_prime_utr_outliers = outlier_indices(three_prime_utr_lengths)

five_prime_utr_ranges_no_outliers = np.delete(np.array(five_prime_utr_ranges), five_prime_utr_outliers, axis=0)
three_prime_utr_ranges_no_outliers = np.delete(np.array(three_prime_utr_ranges), three_prime_utr_outliers, axis=0)

five_prime_utr_lengths_no_outliers = [
    utr_5_range[1] - utr_5_range[0]
    for utr_5_range in five_prime_utr_ranges_no_outliers
]
three_prime_utr_lengths_no_outliers = [
    utr_3_range[1] - utr_3_range[0]
    for utr_3_range in three_prime_utr_ranges_no_outliers
]

_, ax = plt.subplots(figsize=(12, 6))
sns.distplot(dist_to_minus_10, bins=16, kde=False, color='cadetblue')
sns.distplot(dist_to_minus_35, bins=25, kde=False, color='tab:cyan')
ax.axvline(x=-10, ymax=0.05, color='r', linewidth=4)
ax.axvline(x=-35, ymax=0.05, color='r', linewidth=4)
ax.axvline(x=dist_to_minus_10_median, color='cadetblue')
ax.axvline(x=dist_to_minus_35_median, color='tab:cyan')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Distance from TSS (bp)', fontsize=32)
ax.set_ylabel('Count', fontsize=32)
ax.tick_params(axis='both', labelsize='26')
plt.savefig(Path(FIG_PATH, 'promoter_elements.svg'))
plt.show()

_, ax = plt.subplots()
sns.distplot(spacer_lengths, color='deepskyblue', kde=False, bins=np.arange(6, 25))
ax.axvline(x=17, ymax=0.05, color='r', linewidth=4)
ax.axvline(x=np.median(spacer_lengths), color='deepskyblue')
ax.set_xlabel('Distance between -10 and -35 (bp)', fontsize=22)
ax.set_ylabel('Count', fontsize=22)
ax.tick_params(axis='both', labelsize='20')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(Path(FIG_PATH, 'elements_diff.svg'))
plt.show()

_, ax = plt.subplots(figsize=(8, 3))
sns.boxplot(np.array(five_prime_utr_lengths)*-1, showfliers=False, color='royalblue')
ax.set_xlabel("5' UTR Length (bp from start codon)", fontsize=23)
ax.tick_params(axis='both', labelsize='20')
ax.set_xticks([-250, -200, -150, -100, -50, 0])
ax.yaxis.set_ticks_position('none')
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(Path(FIG_PATH, 'five_prime.svg'))
plt.show()

_, ax = plt.subplots(figsize=(8, 3))
sns.boxplot(three_prime_utr_lengths, showfliers=False, color='slateblue')
ax.set_xlabel("3' UTR Length (bp from stop codon)", fontsize=23)
ax.tick_params(axis='both', labelsize='20')
ax.set_xticks([0, 50, 100, 150])
ax.yaxis.set_ticks_position('none')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(Path(FIG_PATH, 'three_prime.svg'))
plt.show()

# want to highlight position of gadW/gadX
gadW = [gene for gene in test_bitome.genes if gene.name == 'gadW'][0]
gadX = [gene for gene in test_bitome.genes if gene.name == 'gadX'][0]
inter_gad_range = gadW.location.end.position, gadX.location.start.position
inter_gad_submat = test_bitome.extract(column_range=inter_gad_range)
inter_gad_bits_per_bp = inter_gad_submat.sum()/inter_gad_submat.shape[1]

bits_per_bp_plot(
    test_bitome,
    [three_prime_utr_ranges_no_outliers, five_prime_utr_ranges_no_outliers, operon_intergenic_ranges, inter_tu_ranges],
    ["3' UTR", "5' UTR", "Intergenic in Operon", 'Inter-TU'],
    kde=False,
    figsize=(10, 15),
    median=True,
    file_path=Path(FIG_PATH, 'bits_per_bp_intergenic')
)

# # --- amino acid and secondary structure ---
# def relative_feature_coverages(
#             feature_name: str,
#             feature_types: List[str],
#             relative: bool = False
#         ) -> Tuple[List[str], List[float]]:
#     """
#     Returns sorted feature types and coverages (as fraction of total base feature coverage)
#
#     :param str feature_name: the base name of the feature
#     :param List[str] feature_types: a list of the types this feature can have
#     :param bool relative: indicates if the coverages should be relative to the whole genome or just the feature
#     :return Tuple[List[str], List[float]] feature_type_coverages: the feature types and their associated coverages,
#     sorted into descending order
#     """
#
#     if relative:
#         total_matrix = test_bitome.extract(row_labels=['amino_acid'], base_name=True)
#         total_zero_inds = np.where(total_matrix.sum(axis=0) == 0)[0]
#         relative_coverage = 1 - (total_zero_inds.shape[0]/len(test_bitome.sequence))
#     else:
#         relative_coverage = 1
#     feature_coverages = []
#     for feature_type in feature_types:
#         feature_matrix = test_bitome.extract(row_labels=[f'{feature_name}_{feature_type}'], base_name=True)
#         zero_indices = np.where(feature_matrix.sum(axis=0) == 0)[0]
#         coverage = 1 - (zero_indices.shape[0]/len(test_bitome.sequence))
#         feature_coverages.append(100*coverage/relative_coverage)
#
#     sorted_types, sorted_coverages = zip(*sorted(zip(feature_types, feature_coverages), key=lambda tup: tup[1]))
#     return sorted_types, sorted_coverages
#
#
# amino_acid_category_lookup = {
#     'L': 'nonpolar',
#     'A': 'nonpolar',
#     'G': 'nonpolar',
#     'V': 'nonpolar',
#     'I': 'nonpolar',
#     'E': 'charged',
#     'S': 'polar',
#     'R': 'charged',
#     'T': 'polar',
#     'D': 'charged',
#     'P': 'nonpolar',
#     'Q': 'polar',
#     'K': 'charged',
#     'F': 'nonpolar',
#     'N': 'polar',
#     'Y': 'polar',
#     'M': 'nonpolar',
#     'H': 'charged',
#     'W': 'nonpolar',
#     'C': 'polar'
# }
# amino_acids = IUPAC.protein.letters
# sorted_aas, sorted_aa_coverages = relative_feature_coverages('amino_acid', amino_acids)
#
# _, ax = plt.subplots()
# x = np.arange(len(sorted_aa_coverages))
# color_lookup = {'nonpolar': 'tab:gray', 'polar': 'tab:blue', 'charged': 'tab:red'}
# colors = [color_lookup[amino_acid_category_lookup[aa]] for aa in sorted_aas]
# plt.barh(x, sorted_aa_coverages, tick_label=sorted_aas, color=colors)
# plt.xlabel('% of Sequence', fontsize=20)
# plt.ylabel('Amino Acid', fontsize=20)
# legend_elems = [Patch(facecolor=col, edgecolor=col, label=lab) for lab, col in color_lookup.items()]
# ax.legend(handles=legend_elems, prop={'size': 16}, loc='lower right')
# ax.tick_params(axis='both', labelsize=15)
# plt.savefig(Path(FIG_PATH, 'amino_acid_coverage.svg'))
# plt.show()
#
# secondary_structures = ['B', 'E', 'I', 'G', 'H', 'T', 'S', '-']
# sorted_ss, sorted_ss_coverages = relative_feature_coverages('secondary_structure', secondary_structures)
# secondary_structure_names = {
#     'B': 'beta bridge',
#     'H': 'alpha helix',
#     'E': 'beta strand',
#     'G': '3-10 helix',
#     'I': 'pi helix',
#     'T': 'turn',
#     'S': 'bend',
#     '-': 'none'
# }
# secondary_structure_groups = {
#     'B': 'beta sheet',
#     'H': 'helix',
#     'E': 'beta sheet',
#     'G': 'helix',
#     'I': 'helix',
#     'T': 'loop',
#     'S': 'loop',
#     '-': 'loop'
# }
# _, ax = plt.subplots()
# x = np.arange(len(sorted_ss_coverages))
# plt.barh(x, sorted_ss_coverages, tick_label=[secondary_structure_names[ss] for ss in sorted_ss])
# plt.xlabel('% of Sequence', fontsize=18)
# ax.tick_params(axis='both', labelsize=16)
# plt.savefig(Path(FIG_PATH, 'ss_coverage.svg'))
# plt.show()
#
# ss_aa_pcts = {}
# for aa, ss in tqdm(product(amino_acids, secondary_structures)):
#     overlap_len = 0
#     ss_len = 0
#     for rf in ['(+1)', '(+2)', '(+3)', '(-1)', '(-2)', '(-3)']:
#         ss_matrix = test_bitome.extract(row_labels=[f'secondary_structure_{ss}_{rf}'], base_name=True)
#         ss_inds = np.where(np.asarray(ss_matrix.sum(axis=0))[0] > 0)[0]
#         aa_matrix = test_bitome.extract(row_labels=[f'amino_acid_{aa}_{rf}'], base_name=True)
#         aa_inds = np.where(np.asarray(aa_matrix.sum(axis=0))[0] > 0)[0]
#         overlap_len += len(set(list(ss_inds)).intersection(set(list(aa_inds))))
#         ss_len += len(ss_inds)
#     ss_group = secondary_structure_groups[ss]
#     ss_group_size = sum([ss_group == ss_group_ref for ss_group_ref in secondary_structure_groups.values()])
#     aa_pct_for_ss = (100*overlap_len/ss_len)/ss_group_size
#     if ss_group in ss_aa_pcts:
#         ss_aa_dict = ss_aa_pcts[ss_group]
#     else:
#         ss_aa_dict = {}
#     if aa in ss_aa_dict:
#         ss_aa_dict[aa] += aa_pct_for_ss
#     else:
#         ss_aa_dict[aa] = aa_pct_for_ss
#     ss_aa_pcts[ss_group] = ss_aa_dict
#
# _, axs = plt.subplots(1, len(ss_aa_pcts), sharey='all', figsize=(14, 7))
# axs = axs.flatten()
# for (ss, aa_pcts_dict), ax in zip(ss_aa_pcts.items(), axs):
#     sorted_aas, sorted_aa_pcts = zip(*sorted(
#         [(aa, pct) for aa, pct in aa_pcts_dict.items()],
#         key=lambda tup: tup[1],
#         reverse=True
#     ))
#     ax.bar(
#         np.arange(len(sorted_aa_pcts)),
#         sorted_aa_pcts,
#         tick_label=sorted_aas,
#         color=[color_lookup[amino_acid_category_lookup[aa]] for aa in sorted_aas]
#     )
#     ax.tick_params(axis='both', labelsize=14)
#     ax.set_title(ss, loc='center', fontsize=24)
# axs[1].set_xlabel('Amino Acid', fontsize=24)
# axs[0].set_ylabel('% with Structure', fontsize=24)
# plt.savefig(Path(FIG_PATH, 'aa_ss.svg'))
# plt.show()
