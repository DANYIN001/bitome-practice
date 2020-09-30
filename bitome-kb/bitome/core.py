# built-in modules
from collections import Counter
import gc
import itertools
from pathlib import Path
import pickle
import sys
from typing import List, Tuple, Union

# third-party modules
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation
from Bio.SeqUtils.CodonUsage import CodonsDict
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.linalg.interpolative import estimate_rank
from scipy.sparse import coo_matrix, csc_matrix, hstack, save_npz, vstack
from scipy.sparse.linalg import aslinearoperator

# local modules
from bitome.genbank import load_genbank_features
from bitome.sbrg import load_i_modulons, load_mrna_structure_ranges, load_ytfs_and_binding_sites
from bitome.regulon_db import load_regulon_db_features
from bitome.utilities import select_features_by_type, signed_int_str

READING_FRAMES = [1, 2, 3, -1, -2, -3]
STRANDS = [1, -1]

# define some commonly-used input/output patterns
PATH_OR_STRING = Union[Path, str]


class Bitome:

    def __init__(self, genbank_file_path: PATH_OR_STRING):
        """
        An object used to construct and store a 'bitome'. A 'bitome' is a feature x position binary matrix of 
        genomic features loaded from source files. Call the load_data method to load genomic data, and the 
        generate_matrix method to construct a sparse matrix representation from that data
        NOTE: this class is currently only set up to load genomic data from specific file formats for the model
        organism E. coli K-12 MG1655

        :param PATH_OR_STRING genbank_file_path: the file path to the GenBank (.gb) file containing genome sequence
        and genomic features to include in this Bitome
        """

        # parse the GenBank record and store the record and some useful features in an attribute
        self.genbank_record = SeqIO.read(genbank_file_path, 'gb')
        self.genbank_id = self.genbank_record.id
        self.sequence = self.genbank_record.seq
        self.description = self.genbank_record.description

        # initialize empty attributes for all the features contained in this Bitome; these will be loaded on request
        # by the load_data and load_matrix methods (heavy)

        # from GenBank record
        self.genes = []
        self.proteins = []
        self._is_k12 = self.genbank_id == 'NC_000913.3'
        self.mobile_elements = []
        self.repeat_regions = []
        self.origin = None
        # hard-code the location of the terminus point if using K-12
        if self._is_k12:
            # from 1. Duggin, I. G. & Bell, S. D. J. Mol. Biol. (2009).
            ter_a = 'AATTAGTATGTTGTAACTAAAGT'
            ter_c = 'ATATAGGATGTTGTAACTAATAT'
            ter_a_start = self.sequence.find(Seq(ter_a).reverse_complement())
            ter_c_start = self.sequence.find(Seq(ter_c))
            self._terminus_location = FeatureLocation(ter_a_start, ter_c_start+len(ter_c), strand=1)
        else:
            self._terminus_location = None

        # from RegulonDB
        self.operons = []
        self.transcription_units = []
        self.promoters = []
        self.terminators = []
        self.attenuators = []
        self.shine_dalgarnos = []
        self.riboswitches = []
        self.transcription_factors = []
        self.tf_binding_sites = []
        self.regulons = []
        self.i_modulons = []

        # properties related to the matrix representation of the information
        self.feature_names = set()
        self._feature_categories = set()
        self.matrix = None
        self.matrix_row_labels = []
        self.matrix_row_categories = []
        self.pre_compressed_matrix = None
        self.pre_compressed_matrix_labels = []
        self.feature_counts = pd.DataFrame(columns=['included', 'total'])
        self._sparsity = None
        self._rank = None
        self._nt_byte_sizes = None
        self._nt_byte_size_counts = None

        # utility/private
        self._matrix_loaded = False

    @classmethod
    def init_from_file(cls, source_file_path: PATH_OR_STRING):
        """
        An alternative constructor that uses a source file name pointing to a pickled Bitome object and creates a new
        instance identical to that previous object; NOTE, using this class method skips the standard __init__ entirely

        :param PATH_OR_STRING source_file_path: the file path of a pickle file containing a dumped Bitome instance
        :return Bitome loaded_bitome: the Bitome object loaded from the provided file
        """

        source_file_path = Path(source_file_path)
        if source_file_path.exists():
            with open(source_file_path, 'rb') as input_file:
                bitome_instance = pickle.load(input_file)
            return bitome_instance
        else:
            raise ValueError('Provided file path does not exist')

    def load_data(self, regulon_db: bool = False):
        """
        Parse raw data files at pre-defined directories to generate objects and links between objects for all
        available features. Currently coded to be specific to E. coli K-12 MG1655 features

        :param bool regulon_db: indicates if genomic features from RegulonDB should be loaded; defaults to False; may
        only be set to true for NC_000913.3 genbank id
        """

        # load knowledgebase feature objects from the GenBank record; assuming that we have a GenBank record if we have
        # constructed this object since that is a required argument for __init__
        self._load_genbank_features()

        if regulon_db:
            if self.genbank_id != 'NC_000913.3':
                raise ValueError(f'RegulonDB is specific for E. coli K12 MG1655 genome annotation NC_000913.3 and '
                                 f'cannot be used with GenBank record id {self.genbank_id}')
            else:
                self._load_regulon_db_features()
                self.i_modulons = load_i_modulons(self.transcription_factors, self.genes)

                # squeeze in SBRG's YTF data; note: load_ytfs_and_binding_sites doesn't return the existing tf objects,
                # but DOES return their new binding sites from yTF data
                ytfs_and_binding_sites = load_ytfs_and_binding_sites(existing_tfs=self.transcription_factors)
                self.transcription_factors += select_features_by_type(ytfs_and_binding_sites, 'transcription_factor')
                self.tf_binding_sites += select_features_by_type(ytfs_and_binding_sites, 'tf_binding_site')

    def _load_genbank_features(self):
        """
        Load knowledgebase feature objects based on a GenBank annotation already loaded to this Bitome; meant to be
        called only by the public load_data method
        """

        all_genbank_features = load_genbank_features(self.genbank_record, terminus=self._terminus_location)
        self.genes = select_features_by_type(all_genbank_features, 'gene')
        self.proteins = select_features_by_type(all_genbank_features, 'protein')
        self.trnas = select_features_by_type(all_genbank_features, 'tRNA')
        self.mobile_elements = select_features_by_type(all_genbank_features, 'mobile_element')
        self.repeat_regions = select_features_by_type(all_genbank_features, 'repeat_region')
        # assumes a single origin of replication; true for NC_000913.3 but maybe not for other genomes
        origin_object = select_features_by_type(all_genbank_features, 'origin')
        if len(origin_object) == 1:
            self.origin = origin_object[0]

    def _load_regulon_db_features(self):
        """
        Load Bitome feature objects based on RegulonDB files stored locally; only to be called from public load_data
        method; only applies for GenBank record NC_000913.3 (E. coli K-12 MG1655)
        """

        # passing the genes to RegulonDB because we will use existing gene objects in lieu of RegulonDB-parsed ones
        # because we're trusting GenBank as our source of gene annotations; note, load_regulon_db_features does NOT
        # return the updated genes (they're already updated in memory)
        all_regulon_db_features = load_regulon_db_features(self.sequence, self.genes)
        self.operons += select_features_by_type(all_regulon_db_features, 'operon')
        self.transcription_units += select_features_by_type(all_regulon_db_features, 'transcription_unit')
        self.promoters += select_features_by_type(all_regulon_db_features, 'promoter')
        self.terminators += select_features_by_type(all_regulon_db_features, 'terminator')
        self.attenuators += select_features_by_type(all_regulon_db_features, 'attenuator')
        self.shine_dalgarnos += select_features_by_type(all_regulon_db_features, 'shine_dalgarno')
        self.riboswitches += select_features_by_type(all_regulon_db_features, 'riboswitch')
        self.transcription_factors += select_features_by_type(all_regulon_db_features, 'transcription_factor')
        self.tf_binding_sites += select_features_by_type(all_regulon_db_features, 'tf_binding_site')
        self.regulons += select_features_by_type(all_regulon_db_features, 'regulon')

    def load_matrix(self):
        """
        Convert a loaded set of bitome features into a sparse bitome matrix. The bitome matrix is a binary matrix
        indicating presence/absence of a given feature in a given genome position. Running this method will populate
        the self.matrix attribute and associated row label lists
        Rows are genomic features, and columns are genomic positions
        """

        # We want to make a coo_matrix here; must send that constructor a list of all non-zero indices
        # however, empirically I've crashed my memory if I compile ALL such indices and make one big call; so I'm
        # splitting the job up and making a distinct sparse matrix for each block of features, then doing some
        # janky manual memory cleanup/garbage collection as I go to ensure I don't bust my laptop's memory

        num_positions = len(self.sequence)

        base_types = ['A', 'T', 'C', 'G']
        base_to_row = {base: row for row, base in enumerate(base_types)}

        base_row_indices = []
        base_r_row_indices = []
        for base, base_r in zip(self.sequence, self.sequence.complement()):
            base_row_indices.append(base_to_row[base])
            base_r_row_indices.append(base_to_row[base_r] + 4)

        # combine the forward and reverse bases into one matrix; double the column index of course
        base_column_indices = np.tile(np.arange(num_positions), 2)
        master_matrix = coo_matrix(
            (
                np.ones(num_positions*2),
                (np.array(base_row_indices + base_r_row_indices), base_column_indices)
            ),
            shape=(8, num_positions), dtype='uint8'
        )
        self.feature_names.add('base')
        master_row_labels = []
        master_row_categories = []
        for base_strand in STRANDS:
            for base_type in base_types:
                master_row_labels.append(f'base_{base_type}_({signed_int_str(base_strand)})')
        master_row_categories += ['core sequence'] * 8

        # hacky memory freeing; delete references to our long lists of base row indices and force garbage collection
        del base_row_indices, base_r_row_indices
        gc.collect()

        def set_up_rows(
                    feat_name: str,
                    feature_types: List[str] = None,
                    reading_frame: bool = True,
                    strand: bool = True
                ) -> Tuple[dict, List[str]]:
            """
            Create a dictionary relating a gene type to a row index for creating a sparse matrix index, and also listing
            of row labels in the order of the indices in the dictionary (not technically index-matched since dicts don't
            have a concept of order)

            :param str feat_name: the base name of the feature
            :param List[str] feature_types: the possible types this feature can have; i.e. ['CDS', 'rRNA'...] for genes
            :param bool reading_frame: indicates if the feature locations are reading frame-sensitive
            :param bool strand: indicates if the feature locations are strand-sensitive; overridden by more specific
            reading_frame=True
            :return Tuple[dict, List[str]] location_dict, row_labels: a dictionary for relating a feature type and
            strand to a row_index, and a list of row_labels in the order of the row indices contained in the dictionary
            """

            if reading_frame:
                frames_or_strands = READING_FRAMES
            elif strand:
                frames_or_strands = STRANDS
            else:
                frames_or_strands = None

            row_lookup = {}
            row_labels = []
            current_row = 0

            if feature_types is None:
                if frames_or_strands is None:
                    row_labels += [feat_name]
                else:
                    for frame_or_strand in frames_or_strands:
                        row_lookup[frame_or_strand] = current_row
                        row_labels.append(feat_name + '_(' + signed_int_str(frame_or_strand) + ')')
                        current_row += 1
            else:
                if frames_or_strands is None:
                    for feature_type in feature_types:
                        row_lookup[feature_type] = current_row
                        row_labels.append(feat_name + '_' + feature_type)
                        current_row += 1
                else:
                    for feature_type in feature_types:
                        type_sub_lookup = {}
                        for frame_or_strand in frames_or_strands:
                            type_sub_lookup[frame_or_strand] = current_row
                            row_name = feat_name + '_' + feature_type + '_(' + signed_int_str(frame_or_strand) + ')'
                            row_labels.append(row_name)
                            current_row += 1
                        row_lookup[feature_type] = type_sub_lookup

            return row_lookup, row_labels

        def add_matrix_feature(feat_ranges: List[Tuple[int, Tuple[int, int]]], row_labels: List[str], category: str):
            """
            Given a listing of feature range indices in the form: (row_index, (col_start, col_end), create a sparse
            matrix and join it to the growing "master" matrix

            :param List[Tuple[int, Tuple[int, int]]] feat_ranges: a list of feature range indices, EACH in the form:
            (row_index, (col_start, col_end)
            :param List[str] row_labels: a list of row labels to add to the master list
            :param str category: the category for this set of feature rows
            """

            all_indices = []
            for row_index, (col_start, col_end) in feat_ranges:
                for col_index in range(col_start, col_end):
                    all_indices.append((row_index, col_index))

            # BIG NOTE: this unique line will wipe out overlapping features if they exist (e.g. hoxC, moxC genes)
            unique_indices = sorted(list(set(all_indices)))
            row_indices = [index_tup[0] for index_tup in unique_indices]
            column_indices = [index_tup[1] for index_tup in unique_indices]

            matrix_shape = len(row_labels), num_positions
            feature_matrix = coo_matrix(
                (np.ones(len(unique_indices)), (row_indices, column_indices)),
                shape=matrix_shape, dtype='uint8'
            )

            nonlocal master_matrix, master_row_labels, master_row_categories
            master_matrix = vstack([master_matrix, feature_matrix])
            master_row_labels += row_labels
            master_row_categories += [category] * len(row_labels)

        gene_feature_name = 'gene'
        self.feature_counts.loc['gene'] = (len(self.included_genes), len(self.genes))
        gene_types = list({gene.gene_type for gene in self.included_genes})
        gene_row_lookup, gene_row_labels = set_up_rows(gene_feature_name, feature_types=gene_types)
        self.feature_names.add(gene_feature_name)

        cogs_feature_name = 'COG'
        genes_with_cog = [gene.cog for gene in self.included_genes if gene.cog is not None]
        self.feature_counts.loc['COG'] = (len(genes_with_cog), len(self.genes))
        cogs_types = list(set(genes_with_cog))
        cogs_row_lookup, cogs_row_labels = set_up_rows(cogs_feature_name, feature_types=cogs_types)
        self.feature_names.add(cogs_feature_name)

        gene_ranges = []
        cogs_ranges = []
        for gene in self.included_genes:

            gene_type = gene.gene_type
            gene_reading_frame = gene.reading_frame

            # Gene objects have support for CompoundLocations (this is all pseudo genes, both single-location
            # FeatureLocation and multi-location CompoundLocation objects have a parts attribute with 1 or 2+
            # FeatureLocation objects, respectively
            gene_locations = gene.location.parts
            for gene_sublocation in gene_locations:
                gene_start = gene_sublocation.start.position
                gene_end = gene_sublocation.end.position
                gene_row_index = gene_row_lookup[gene_type][gene_reading_frame]
                gene_ranges.append((gene_row_index, (gene_start, gene_end)))

                gene_cog = gene.cog
                if gene_cog is not None:
                    cogs_row_index = cogs_row_lookup[gene_cog][gene_reading_frame]
                    cogs_ranges.append((cogs_row_index, (gene_start, gene_end)))

        add_matrix_feature(gene_ranges, gene_row_labels, 'core sequence')
        add_matrix_feature(cogs_ranges, cogs_row_labels, 'computed')
        del gene_ranges, cogs_ranges
        gc.collect()

        codon_feature_name = 'codon'
        codon_types = list({''.join(codon) for codon in itertools.product(['A', 'C', 'T', 'G'], repeat=3)})
        codon_row_lookup, codon_row_labels = set_up_rows(codon_feature_name, feature_types=codon_types)
        self.feature_names.add(codon_feature_name)

        codon_ranges = []
        for coding_gene in self.coding_genes:
            coding_gene_sequence = str(coding_gene.sequence)
            codons = [coding_gene_sequence[i:i+3] for i in range(0, len(coding_gene_sequence), 3)]

            # codons are in coding order but location is absolute; swap the codons if reverse strand feature
            coding_gene_rf = coding_gene.reading_frame
            if coding_gene_rf < 0:
                codons.reverse()

            coding_gene_left = coding_gene.location.start.position
            coding_gene_right = coding_gene.location.end.position

            for codon_left, codon in zip(range(coding_gene_left, coding_gene_right, 3), codons):
                codon_row_index = codon_row_lookup[codon][coding_gene_rf]
                codon_ranges.append((codon_row_index, (codon_left, codon_left+3)))

        add_matrix_feature(codon_ranges, codon_row_labels, 'core sequence')
        del codon_ranges
        gc.collect()

        # only one 'type' of protein; use a placeholder as the type; we will make sure later that this doesn't
        # become a suffix for the feature row label
        protein_feature_name = 'protein'
        protein_row_lookup, protein_row_labels = set_up_rows(protein_feature_name)
        self.feature_names.add(protein_feature_name)

        # sneak a 'U' in here for selenocysteine, a couple of proteins have that in K12
        amino_acid_feature_name = 'amino_acid'
        if self._is_k12:
            amino_acid_types = list(IUPAC.protein.letters + 'U')
        else:
            amino_acid_types = list(IUPAC.protein.letters)
        aa_row_lookup, aa_row_labels = set_up_rows(amino_acid_feature_name, feature_types=amino_acid_types)
        self.feature_names.add(amino_acid_feature_name)

        self.feature_counts.loc['protein'] = (len(self.included_proteins), len(self.proteins))

        exposure_count = len([prot for prot in self.included_proteins if prot.amino_acid_exposures is not None])
        self.feature_counts.loc['residue_exposure'] = (exposure_count, len(self.included_proteins))

        ss_count = len([prot for prot in self.included_proteins if prot.secondary_structure is not None])
        self.feature_counts.loc['secondary_structure'] = (ss_count, len(self.included_proteins))

        if exposure_count > 0:
            residue_exposure_feature_name = 'residue_exposure'
            residue_exposures = ['e', '-']
            exposure_row_lookup, exposure_row_labels = set_up_rows(
                residue_exposure_feature_name,
                feature_types=residue_exposures
            )
            self.feature_names.add(residue_exposure_feature_name)
        else:
            exposure_row_lookup, exposure_row_labels = {}, []

        # prediction from DSSP program; go here for meanings:
        # http://www.csb.yale.edu/userguides/databases/dssp/dssp_man.html
        if ss_count > 0:
            ss_feature_name = 'secondary_structure'
            secondary_structures = ['B', 'E', 'I', 'G', 'H', 'T', 'S', '-']
            ss_row_lookup, ss_row_labels = set_up_rows(ss_feature_name, feature_types=secondary_structures)
            self.feature_names.add(ss_feature_name)
        else:
            ss_row_lookup, ss_row_labels = {}, []

        protein_ranges = []
        aa_ranges = []
        exposure_ranges = []
        ss_ranges = []

        for protein in self.included_proteins:

            protein_start = protein.location.start.position
            protein_end = protein.location.end.position
            protein_reading_frame = protein.reading_frame
            protein_row_index = protein_row_lookup[protein_reading_frame]
            protein_ranges.append((protein_row_index, (protein_start, protein_end)))

            def add_residue_feature(feat_row_lookup: dict, range_list: list, protein_feature_sequence: str):
                """
                For a protein feature annotated by residue, add it to the OUTER SCOPE dictionary for that feature

                :param dict feat_row_lookup: a dict of feature types/reading frames to row_indices for this feature
                :param list range_list: the list of ranges this feature occupies; meant to modify the OUTER version
                :param str protein_feature_sequence: the actual sequence of this feature's values for this protein
                """
                if protein_feature_sequence is None:
                    return
                feature_sequence_list = list(protein_feature_sequence)
                if protein_reading_frame < 0:
                    feature_sequence_list.reverse()
                for feature_left, feature_type in zip(range(protein_start, protein_end, 3), feature_sequence_list):
                    feat_row_index = feat_row_lookup[feature_type][protein_reading_frame]
                    range_list.append((feat_row_index, (feature_left, feature_left+3)))

            add_residue_feature(aa_row_lookup, aa_ranges, protein.amino_acid_sequence)
            add_residue_feature(exposure_row_lookup, exposure_ranges, protein.amino_acid_exposures)
            add_residue_feature(ss_row_lookup, ss_ranges, protein.secondary_structure)

        add_matrix_feature(protein_ranges, protein_row_labels, 'core sequence')
        add_matrix_feature(aa_ranges, aa_row_labels, 'core sequence')
        add_matrix_feature(exposure_ranges, exposure_row_labels, 'computed')
        add_matrix_feature(ss_ranges, ss_row_labels, 'computed')
        del protein_ranges, aa_ranges, exposure_ranges, ss_ranges
        gc.collect()

        if self.origin is not None:
            origin_feature_name = 'origin'
            self.feature_counts.loc['origin'] = (1, 1)
            ori_row_lookup, ori_row_labels = set_up_rows(origin_feature_name, reading_frame=False, strand=False)
            ori_ranges = [(0, (self.origin.location.start.position, self.origin.location.end.position))]
            self.feature_names.add(origin_feature_name)
            add_matrix_feature(ori_ranges, ori_row_labels, 'core sequence')

            del ori_ranges
            gc.collect()

        # for the RegulonDB features, only want to use those that are linked to a Gene
        tus_to_use = [tu for tu in self.transcription_units if tu.genes]
        self.feature_counts.loc['TU'] = (len(tus_to_use), len(self.transcription_units))

        operons_to_use = [tu.operon for tu in tus_to_use]
        self.feature_counts.loc['operon'] = (len(operons_to_use), len(self.operons))

        promoters_to_use = [tu.promoter for tu in tus_to_use if tu.promoter is not None]
        self.feature_counts.loc['promoter'] = (len(promoters_to_use), len(self.promoters))

        terminators_w_gene = [term for term in self.terminators if term.transcription_unit in tus_to_use]
        terminators_to_use = [term for term in terminators_w_gene if term.terminator_class == 'rho-independent']
        terminators_rho_to_use = [term for term in terminators_w_gene if term.terminator_class == 'rho-dependent']
        self.feature_counts.loc['terminator'] = (
            len(terminators_to_use + terminators_rho_to_use),
            len(self.terminators)
        )

        attenuators_w_gene = [att for att in self.attenuators if att.gene in self.included_genes]
        attenuators_transcrip_to_use = [att for att in attenuators_w_gene if att.attenuator_type == 'Transcriptional']
        attenuators_translat_to_use = [att for att in attenuators_w_gene if att.attenuator_type == 'Translational']
        self.feature_counts.loc['attenuator'] = (
            len(attenuators_transcrip_to_use + attenuators_translat_to_use),
            len(self.attenuators)
        )

        shine_dalgarnos_to_use = [sd for sd in self.shine_dalgarnos if sd.gene in self.included_genes]
        self.feature_counts.loc['Shine-Dalgarno'] = (
            len(shine_dalgarnos_to_use),
            len(self.shine_dalgarnos)
        )

        riboswitches_to_use = [rs for rs in self.riboswitches if rs.gene in self.included_genes]
        self.feature_counts.loc['riboswitch'] = (
            len(riboswitches_to_use),
            len(self.riboswitches)
        )

        # ALL of these standard types can be added to the master dict in the exact same fashion; zip-loop through them
        operon_feature_names = [
            'mobile_element', 'repeat_region', 'TU', 'operon', 'promoter', 'terminator',
            'terminator_rho', 'attenuator_transcription', 'attenuator_translation', 'Shine-Dalgarno',
            'riboswitch'
        ]
        operon_feature_lists = [
            self.mobile_elements, self.repeat_regions, tus_to_use, operons_to_use, promoters_to_use, terminators_to_use,
            terminators_rho_to_use, attenuators_transcrip_to_use, attenuators_translat_to_use, shine_dalgarnos_to_use,
            riboswitches_to_use
        ]
        self.feature_counts.loc['mobile_element'] = (len(self.mobile_elements), len(self.mobile_elements))
        self.feature_counts.loc['repeat_region'] = (len(self.repeat_regions), len(self.repeat_regions))

        for feature_name, features in zip(operon_feature_names, operon_feature_lists):

            if not features:
                continue

            # these operon/misc GenBank features are all strand-aware but reading-frame agnostic
            feature_row_lookup, feature_row_labels = set_up_rows(feature_name, reading_frame=False)
            self.feature_names.add(feature_name)

            feature_ranges = []
            for feature in features:
                feature_start = feature.location.start.position
                feature_end = feature.location.end.position
                feature_strand = feature.location.strand
                feature_row_index = feature_row_lookup[feature_strand]
                feature_ranges.append((feature_row_index, (feature_start, feature_end)))

            add_matrix_feature(feature_ranges, feature_row_labels, 'hybrid')
            del feature_ranges

        gc.collect()

        if promoters_to_use:
            box_10_feature_name = '-10_box'
            box_10_row_lookup, box_10_labels = set_up_rows(box_10_feature_name, reading_frame=False)
            box_10_ranges = []
            box_35_feature_name = '-35_box'
            box_35_row_lookup, box_35_labels = set_up_rows(box_35_feature_name, reading_frame=False)
            box_35_ranges = []
            self.feature_names |= {box_10_feature_name, box_35_feature_name}

            promoter_box_count = len([prom for prom in promoters_to_use if prom.box_10_location is not None])
            self.feature_counts.loc['-10_box'] = (promoter_box_count, len(promoters_to_use))
            self.feature_counts.loc['-35_box'] = (promoter_box_count, len(promoters_to_use))

            for promoter in promoters_to_use:

                promoter_strand = promoter.location.strand
                box_10_location = promoter.box_10_location
                box_35_location = promoter.box_35_location

                # assume that either both or none are populated
                if box_10_location is not None and box_35_location is not None:
                    box_10_row = box_10_row_lookup[promoter_strand]
                    box_10_ranges.append((box_10_row, (box_10_location.start.position, box_10_location.end.position)))
                    box_35_row = box_35_row_lookup[promoter_strand]
                    box_35_ranges.append((box_35_row, (box_35_location.start.position, box_35_location.end.position)))

            add_matrix_feature(box_10_ranges, box_10_labels, 'hybrid')
            add_matrix_feature(box_35_ranges, box_35_labels, 'hybrid')

            del box_10_ranges, box_35_ranges
            gc.collect()

            tss_feature_name = 'TSS'
            tss_row_lookup, tss_row_labels = set_up_rows(tss_feature_name, reading_frame=False)
            self.feature_names.add(tss_feature_name)

            tss_ranges = []
            for promoter in promoters_to_use:
                tss_location = promoter.tss_location
                promoter_strand = promoter.location.strand
                tss_row_index = tss_row_lookup[promoter_strand]
                tss_ranges.append((tss_row_index, (tss_location.start.position, tss_location.end.position)))

            add_matrix_feature(tss_ranges, tss_row_labels, 'hybrid')
            del tss_ranges
            gc.collect()

        if tus_to_use:
            tts_feature_name = 'TTS'
            tts_row_lookup, tts_row_labels = set_up_rows(tts_feature_name, reading_frame=False)
            self.feature_names.add(tts_feature_name)

            tts_ranges = []
            for tu in tus_to_use:
                tu_strand = tu.location.strand
                tts_row_index = tts_row_lookup[tu_strand]
                # TTS is 1-indexed, but the ranges are 0-indexed
                tts_ranges.append((tts_row_index, (tu.tts-1, tu.tts)))

            add_matrix_feature(tts_ranges, tts_row_labels, 'hybrid')
            del tts_ranges
            gc.collect()

        # define sigmulons, regulons, and i-modulons based on the transcription units they regulate
        # only include transcription units/promoters/genes that we have included previously
        sigmulon_types = set()
        for promoter in promoters_to_use:
            sigmulon_types |= set(promoter.sigma_factors)
        sigmulon_feature_name = 'sigmulon'
        sigmulon_row_lookup, sigmulon_row_labels = set_up_rows(
            sigmulon_feature_name, feature_types=list(sigmulon_types), reading_frame=False, strand=False
        )
        self.feature_names.add(sigmulon_feature_name)

        promoter_w_sigmulon_count = 0

        sigmulon_ranges = []
        for promoter in promoters_to_use:
            promoter_tu = promoter.transcription_unit
            tu_start = promoter_tu.location.start.position
            tu_end = promoter_tu.location.end.position
            if promoter.sigma_factors:
                promoter_w_sigmulon_count += 1
            for sigma_factor in promoter.sigma_factors:
                sigmulon_row_index = sigmulon_row_lookup[sigma_factor]
                sigmulon_ranges.append((sigmulon_row_index, (tu_start, tu_end)))

        self.feature_counts.loc['sigmulon'] = (promoter_w_sigmulon_count, len(promoters_to_use))
        add_matrix_feature(sigmulon_ranges, sigmulon_row_labels, 'hybrid')

        i_modulon_types = []
        for i_modulon in self.i_modulons:
            if list(set(i_modulon.genes).intersection(self.included_genes)):
                i_modulon_types.append(i_modulon.name)
        i_modulon_feature_name = 'i-modulon'
        i_modulon_row_lookup, i_modulon_row_labels = set_up_rows(
            i_modulon_feature_name, feature_types=i_modulon_types, reading_frame=False, strand=False
        )
        self.feature_names.add(i_modulon_feature_name)
        i_modulon_ranges = []
        all_i_modulon_genes = set()
        i_modulon_count = 0
        for i_modulon in self.i_modulons:
            # only include genes from the i-modulon that we're overall including
            i_modulon_genes = list(set(i_modulon.genes).intersection(self.included_genes))
            all_i_modulon_genes |= set(i_modulon_genes)
            if i_modulon_genes:
                i_modulon_count += 1
                i_modulon_row_index = i_modulon_row_lookup[i_modulon.name]
                for i_mod_gene in i_modulon_genes:
                    gene_start = i_mod_gene.location.start.position
                    gene_end = i_mod_gene.location.end.position
                    i_modulon_ranges.append((i_modulon_row_index, (gene_start, gene_end)))

        self.feature_counts.loc['i-modulon'] = (i_modulon_count, len(self.i_modulons))
        self.feature_counts.loc['i-modulon_genes'] = (len(all_i_modulon_genes), len(self.included_genes))
        add_matrix_feature(i_modulon_ranges, i_modulon_row_labels, 'computed')

        # consider distinct regulatory functions of regulons as distinct types
        regulon_types = []
        for regulon in self.regulons:
            for reg_func, reg_promoters in list(regulon.regulated_promoters.items()):
                if list(set(reg_promoters).intersection(set(promoters_to_use))):
                    regulon_types.append(regulon.name + '_' + reg_func)
        regulon_feature_name = 'regulon'
        regulon_row_lookup, regulon_row_labels = set_up_rows(
            regulon_feature_name, feature_types=regulon_types, reading_frame=False, strand=False
        )
        self.feature_names.add(regulon_feature_name)
        regulon_ranges = []
        regulon_func_total_count = 0
        regulon_func_count = 0
        all_regulon_promoters = set()
        for regulon in self.regulons:

            for regulon_func, reg_promoters in regulon.regulated_promoters.items():
                regulon_func_total_count += 1
                reg_promoters_to_use = list(set(reg_promoters).intersection(set(promoters_to_use)))
                if reg_promoters_to_use:
                    regulon_func_count += 1
                    regulon_func_row = regulon_row_lookup[regulon.name + '_' + regulon_func]
                    all_regulon_promoters |= set(reg_promoters_to_use)
                    for reg_promoter in reg_promoters_to_use:
                        promoter_tu = reg_promoter.transcription_unit
                        tu_start = promoter_tu.location.start.position
                        tu_end = promoter_tu.location.end.position
                        regulon_ranges.append((regulon_func_row, (tu_start, tu_end)))

        self.feature_counts.loc['regulon'] = (regulon_func_count, regulon_func_total_count)
        self.feature_counts.loc['regulon_promoters'] = (len(all_regulon_promoters), len(promoters_to_use))
        add_matrix_feature(regulon_ranges, regulon_row_labels, 'hybrid')

        del sigmulon_ranges, i_modulon_ranges, regulon_ranges
        gc.collect()

        tfbs_names = []
        for tf in self.transcription_factors:
            for tf_final_state, sites in tf.binding_sites.items():
                if len(sites) > 0:
                    tfbs_names.append(tf_final_state)
        tfbs_feature_name = 'TFBS'
        tfbs_row_lookup, tfbs_row_labels = set_up_rows(tfbs_feature_name, tfbs_names, reading_frame=False, strand=False)
        self.feature_names.add(tfbs_feature_name)
        tfbs_ranges = []
        tf_count_total = 0
        tf_count_used = 0
        site_count = 0
        for tf in self.transcription_factors:
            for tf_final_state, sites in tf.binding_sites.items():
                tf_count_total += 1
                if sites:
                    tf_count_used += 1
                    site_count += len(sites)
                tfbs_row_index = tfbs_row_lookup.get(tf_final_state)
                for site in sites:
                    tfbs_start = site.location.start.position
                    tfbs_end = site.location.end.position
                    tfbs_ranges.append((tfbs_row_index, (tfbs_start, tfbs_end)))

        self.feature_counts.loc['TF'] = (tf_count_used, tf_count_total)
        self.feature_counts.loc['TFBS'] = (site_count, site_count)

        add_matrix_feature(tfbs_ranges, tfbs_row_labels, 'hybrid')
        del tfbs_ranges
        gc.collect()

        # the mRNA folding energy data is not stored in self; file parser returns ranges of mRNA structure
        if self.genbank_id == 'NC_000913.3':
            mrna_structure_feature_name = 'mRNA_structure'
            mrna_structure_row_lookup, mrna_structure_row_labels = set_up_rows(
                mrna_structure_feature_name,
                reading_frame=False,
                strand=False
            )
            self.feature_names.add(mrna_structure_feature_name)
            mrna_structure_ranges = load_mrna_structure_ranges()
            mrna_structure_ranges = [(0, tup) for tup in mrna_structure_ranges]
            self.feature_counts.loc['mRNA_structure'] = (len(mrna_structure_ranges), len(mrna_structure_ranges))
            add_matrix_feature(mrna_structure_ranges, mrna_structure_row_labels, 'computed')

            del mrna_structure_ranges
            gc.collect()

        # finally, hacky way to remove strand information for a single-stranded genome (e.g. coronavirus)
        single_stranded = self.genbank_id == 'NC_045512.2'
        if single_stranded:
            reverse_strand_indices = [i for i, row_label in enumerate(master_row_labels) if '(-' in row_label]
            indices_to_keep = np.array(list(set(range(master_matrix.shape[0])).difference(set(reverse_strand_indices))))
            master_matrix = master_matrix.tocsr()[indices_to_keep, :]
            master_row_labels = list(np.array(master_row_labels)[indices_to_keep])
            master_row_categories = list(np.array(master_row_categories)[indices_to_keep])

        self.matrix = master_matrix.tocsc()
        self.matrix_row_labels = master_row_labels
        self.matrix_row_categories = master_row_categories
        self._matrix_loaded = True

    def load_pre_compressed_matrix(self):
        """
        Convert a loaded set of bitome features into a sparse, pre-compressed bitome matrix. This matrix is NOT binary;
        by pre-compressed, we mean that features with multiple categories (e.g. base pair, with A, C, G, and T) are
        encoded in a single row with, say, the integers 0, 1, 2, 3. Care is taken to ensure that integers are not
        overloaded for multiple different features (i.e. A for adenine and A for alanine in a protein sequence should
        not be conflated)

        Rows are genomic features, and columns are genomic positions

        Contains some redundant code with the load_matrix method for features that are not "compressible" in this way;
        these include mere presence/absence features and heavily overlapping features such as TF binding sites, regulon,
        sigmulon, i-modulon
        """

        def build_sub_matrix(ranges: List[Tuple[int, int, Tuple[int, int]]]) -> coo_matrix:
            """
            Given a list of range tuples to annotate in a pre-compressed matrix in the form:

            (
                int_code_for_type,
                row_index (i.e. for reading frame or strand),
                (col_start, col_end)
            )

            Return a coo matrix representing all of the ranges
            """

            all_data_and_indices = []
            for type_int_code, row_index, (col_start, col_end) in ranges:
                for col_index in range(col_start, col_end):
                    all_data_and_indices.append((type_int_code, row_index, col_index))

            # BIG NOTE: this unique line will wipe out overlapping features if they exist (e.g. hoxC, moxC genes)
            unique_indices = sorted(list(set(all_data_and_indices)))
            type_ints, row_indices, column_indices = zip(*unique_indices)

            matrix_shape = max(row_indices)+1, num_positions
            return coo_matrix(
                (type_ints, (row_indices, column_indices)),
                shape=matrix_shape, dtype='uint8'
            )

        # set useful constants and initial values
        int_code_max = 0
        num_positions = len(self.sequence)
        reading_frame_to_row = {frame: row for row, frame in enumerate(READING_FRAMES)}

        # start with the bases, encoding
        base_types = ['A', 'C', 'T', 'G']
        base_to_int = {base: int_code for int_code, base in enumerate(base_types)}
        int_code_max += len(base_types)
        pre_compressed_matrix = coo_matrix(
            (
                np.array([base_to_int[base] for base in self.sequence]),
                (np.zeros(num_positions), np.arange(num_positions))
            ),
            shape=(1, num_positions), dtype='uint8'
        )
        pre_compressed_row_labels = ['base']

        gene_types = list({gene.gene_type for gene in self.included_genes})
        gene_type_to_int = {gene_type: int_code + int_code_max for int_code, gene_type in enumerate(gene_types)}
        int_code_max += len(gene_types)

        cog_types = list({gene.cog for gene in self.included_genes if gene.cog is not None})
        cog_type_to_int = {cog_type: int_code + int_code_max for int_code, cog_type in enumerate(cog_types)}
        int_code_max += len(cog_types)

        # range tuples in the form (type_int, row, (col_start, col_end))
        gene_ranges = []
        cogs_ranges = []
        for gene in self.included_genes:

            gene_type = gene.gene_type
            gene_type_int = gene_type_to_int[gene_type]

            # Gene objects have support for CompoundLocations (this is all pseudo genes, both single-location
            # FeatureLocation and multi-location CompoundLocation objects have a parts attribute with 1 or 2+
            # FeatureLocation objects, respectively
            gene_locations = gene.location.parts
            for gene_sublocation in gene_locations:
                gene_start = gene_sublocation.start.position
                gene_end = gene_sublocation.end.position
                gene_row_index = reading_frame_to_row[gene.reading_frame]
                gene_ranges.append((gene_type_int, gene_row_index, (gene_start, gene_end)))

                if gene.cog is not None:
                    cog_type_int = cog_type_to_int[gene.cog]
                    cogs_ranges.append((cog_type_int, gene_row_index, (gene_start, gene_end)))

        gene_matrix = build_sub_matrix(gene_ranges)
        cog_matrix = build_sub_matrix(cogs_ranges)
        pre_compressed_matrix = vstack([pre_compressed_matrix, gene_matrix, cog_matrix])
        pre_compressed_row_labels += [f'gene_({signed_int_str(rf)})' for rf in READING_FRAMES]
        pre_compressed_row_labels += [f'COG_({signed_int_str(rf)})' for rf in READING_FRAMES]

        del gene_ranges, cogs_ranges
        gc.collect()

        codon_types = list({''.join(codon) for codon in itertools.product(['A', 'C', 'T', 'G'], repeat=3)})
        codon_type_to_int = {codon_type: int_code + int_code_max for int_code, codon_type in enumerate(codon_types)}
        int_code_max += len(codon_types)

        codon_ranges = []
        for coding_gene in self.coding_genes:
            coding_gene_sequence = str(coding_gene.sequence)
            codons = [coding_gene_sequence[i:i+3] for i in range(0, len(coding_gene_sequence), 3)]

            # codons are in coding order but location is absolute; swap the codons if reverse strand feature
            if coding_gene.reading_frame < 0:
                codons.reverse()

            coding_gene_left = coding_gene.location.start.position
            coding_gene_right = coding_gene.location.end.position
            codon_row_index = reading_frame_to_row[coding_gene.reading_frame]

            for codon_left, codon in zip(range(coding_gene_left, coding_gene_right, 3), codons):
                codon_type_int = codon_type_to_int[codon]
                codon_ranges.append((codon_type_int, codon_row_index, (codon_left, codon_left+3)))

        codon_matrix = build_sub_matrix(codon_ranges)
        pre_compressed_matrix = vstack([pre_compressed_matrix, codon_matrix])
        pre_compressed_row_labels += [f'codon_({signed_int_str(rf)})' for rf in READING_FRAMES]

        del codon_ranges
        gc.collect()

        # sneak a 'U' in here for selenocysteine, a couple of proteins have that
        amino_acid_types = list(IUPAC.protein.letters + 'U')
        aa_type_to_int = {aa_type: int_code + int_code_max for int_code, aa_type in enumerate(amino_acid_types)}
        int_code_max += len(amino_acid_types)

        residue_exposures = ['e', '-']
        exposure_to_int = {exp_type: int_code + int_code_max for int_code, exp_type in enumerate(residue_exposures)}
        int_code_max += len(residue_exposures)

        secondary_structures = ['B', 'E', 'I', 'G', 'H', 'T', 'S', '-']
        ss_type_to_int = {ss_type: int_code + int_code_max for int_code, ss_type in enumerate(secondary_structures)}
        int_code_max += len(secondary_structures)

        aa_ranges = []
        exposure_ranges = []
        ss_ranges = []
        for protein in self.included_proteins:

            protein_start = protein.location.start.position
            protein_end = protein.location.end.position

            def add_residue_feature(feat_type_int_lookup: dict, range_list: list, protein_feature_sequence: str):
                """
                For a protein feature annotated by residue, add a pre-compressed-style range after looking up that
                feature type's integer code; handles cases where the protein is missing the feature entirely

                :param dict feat_type_int_lookup: a dict of feature types to integer codes for this feature
                :param list range_list: the list of ranges this feature occupies; meant to modify the OUTER version
                :param str protein_feature_sequence: the actual sequence of this feature's values for this protein
                """
                if protein_feature_sequence is None:
                    return
                feature_sequence_list = list(protein_feature_sequence)
                if protein.reading_frame < 0:
                    feature_sequence_list.reverse()
                for feature_left, feature_type in zip(range(protein_start, protein_end, 3), feature_sequence_list):
                    feat_type_int = feat_type_int_lookup[feature_type]
                    feat_row_index = reading_frame_to_row[protein.reading_frame]
                    range_list.append((feat_type_int, feat_row_index, (feature_left, feature_left+3)))

            add_residue_feature(aa_type_to_int, aa_ranges, protein.amino_acid_sequence)
            add_residue_feature(exposure_to_int, exposure_ranges, protein.amino_acid_exposures)
            add_residue_feature(ss_type_to_int, ss_ranges, protein.secondary_structure)

        aa_matrix = build_sub_matrix(aa_ranges)
        exp_matrix = build_sub_matrix(exposure_ranges)
        ss_matrix = build_sub_matrix(ss_ranges)
        pre_compressed_matrix = vstack([pre_compressed_matrix, aa_matrix, exp_matrix, ss_matrix])
        pre_compressed_row_labels += [f'amino_acid_({signed_int_str(rf)})' for rf in READING_FRAMES]
        pre_compressed_row_labels += [f'residue_exposure_({signed_int_str(rf)})' for rf in READING_FRAMES]
        pre_compressed_row_labels += [f'secondary_structure_({signed_int_str(rf)})' for rf in READING_FRAMES]

        del aa_ranges, exposure_ranges, ss_ranges
        gc.collect()

        # now we will finish this off in a slightly hacky way; we've hand-picked the rows (above) that CAN be "pre-
        # compressed; now we are going to load up the original matrix, strip off the rows that we've pre-compressed
        # here, and stitch everything together
        if not self._matrix_loaded:
            self.load_matrix()
        binary_matrix = self.matrix

        labels_to_remove = ['base', 'gene', 'COG', 'codon', 'amino_acid', 'residue_exposure', 'secondary_structure']
        row_indices_to_keep = [
            i for i, row_label in enumerate(self.matrix_row_labels)
            if not any(remove_label in row_label for remove_label in labels_to_remove)
        ]
        stripped_binary_matrix = binary_matrix[row_indices_to_keep, :]
        stripped_binary_row_labels = [self.matrix_row_labels[keep_index] for keep_index in row_indices_to_keep]

        pre_compressed_matrix = vstack([pre_compressed_matrix, stripped_binary_matrix])
        pre_compressed_row_labels = pre_compressed_row_labels + stripped_binary_row_labels

        self.pre_compressed_matrix = pre_compressed_matrix.tocsc()
        self.pre_compressed_matrix_labels = pre_compressed_row_labels

    def validate_matrix(self) -> Union[bool, None]:
        """
        Checks the validity of the matrix construction by performing a battery of QC/QA tests

        :return bool validity: indicates if the matrix passed all of the QC/QA checks
        """

        if not self._matrix_loaded:
            print('Matrix not loaded yet. Please run load_matrix method before checking matrix validity')
            return

        validity = True

        # --- TEST: GC/AT row fractions add to 1 ---
        at_matrix = self.extract(row_labels=['base_A', 'base_T'], base_name=True)
        gc_matrix = self.extract(row_labels=['base_G', 'base_C'], base_name=True)
        matrix_length = self.matrix.shape[1]
        # since the base sub-matrices include forward and reverse, these fractions are double the single strand values
        matrix_at_fraction = (at_matrix.sum()/matrix_length)/2
        matrix_gc_fraction = (gc_matrix.sum()/matrix_length)/2
        if matrix_at_fraction + matrix_gc_fraction != 1.0:
            print('Matrix AT and GC fractions do not add to 1')
            validity = False

        # --- TEST: GC fraction of sequence computed from sequence/matrix are same
        sequence_length = len(self.sequence)
        sequence_gc_fraction = (self.sequence.count('G') + self.sequence.count('C')) / sequence_length
        if sequence_gc_fraction != matrix_gc_fraction:
            print('Matrix GC content does not match GC content calculated directly from reference sequence')
            validity = False

        # --- TEST: GC content calculated from matrix matches literature value
        # SOURCE: https://bionumbers.hms.harvard.edu/bionumber.aspx?id=100528&ver=6&trm=e+coli+k12+gc+content&org=
        if round(matrix_gc_fraction, 3) != 0.508:
            print('Matrix GC content does not match literature value of 50.8% for E. coli K-12')
            validity = False

        # --- TEST: no features that have 0 annotated locations ---
        for base_row_name in self.feature_names:
            base_row_sub_mat = self.extract(row_labels=[base_row_name], base_name=True)
            if base_row_sub_mat.sum() == 0:
                print(f'{base_row_name}: No features present!')
                validity = False

        # --- TEST: amino acid rows and the codon rows that code for them should match ---
        codon_table = CodonTable.generic_by_id[11]
        start_codons = codon_table.start_codons
        # TODO don't skip U even though it's super annoying and the NCBI codon table doesn't know about it
        for amino_acid in list(IUPAC.protein.letters):

            amino_acid_sub_mat = self.extract(row_labels=['amino_acid_' + amino_acid], base_name=True)
            aa_sum = amino_acid_sub_mat.sum()

            aa_codons = [codon for codon, aa in codon_table.forward_table.items() if aa == amino_acid]
            aa_codon_sub_mat = self.extract(row_labels=['codon_' + codon for codon in aa_codons], base_name=True)
            aa_codon_sum = aa_codon_sub_mat.sum()

            # need to handle a wild edge case; codons serve multiple purposes if they are start codons; a codon
            # that codes I in the middle of a transcript can code M if its the start; so we want to modify the aa
            # codon sum by subtracting cases where we've actually over-counted a given amino acids codons because a
            # start codon appeared; and Methionine is the opposite, need to add in codons that are not M
            aa_and_start_codons = list(set(start_codons).intersection(set(aa_codons)))
            if amino_acid == 'M':
                non_m_start_codons = list(set(start_codons).difference(set(aa_codons)))
                for gene in [gene for gene in self.genes if gene.gene_type == 'CDS' and gene.transcription_units]:
                    if gene.sequence[:3] in non_m_start_codons:
                        aa_codon_sum += 3

            elif len(aa_and_start_codons) > 0:
                for gene in [gene for gene in self.genes if gene.gene_type == 'CDS' and gene.transcription_units]:
                    if gene.sequence[:3] in aa_and_start_codons:
                        aa_codon_sum -= 3

            if aa_sum != aa_codon_sum:
                print(amino_acid + ': codons for this amino acid do not add to same value')
                validity = False

        # --- TEST: coding gene, protein, codon, amino acid, protein property rows divisible by 3
        coding_related_features = [
            'gene_CDS', 'codon', 'protein', 'amino_acid', 'residue_exposure', 'secondary_structure'
        ]
        for coding_feature_name in coding_related_features:
            feature_sub_mat = self.extract(row_labels=[coding_feature_name], base_name=True)
            if feature_sub_mat.sum() % 3 != 0:
                print(coding_feature_name + ': coding feature sum expected to be divisible by 3')
                validity = False

        # --- TEST: codon row sums (divided by 3) should match codons counted gene-by-gene (from genes in bitome)
        codon_count_matrix = {}
        codons = list({''.join(codon) for codon in itertools.product(['A', 'C', 'T', 'G'], repeat=3)})
        for codon in codons:
            codon_submatrix = self.extract(row_labels=['codon_' + codon], base_name=True)
            codon_count = int(codon_submatrix.sum()/3)
            codon_count_matrix[codon] = codon_count

        codon_count_genes = CodonsDict.copy()
        genes_to_count = [
            gene for gene in self.genes if gene.gene_type == 'CDS' and gene.transcription_units
            and gene.protein is not None
        ]
        for gene_to_count in genes_to_count:

            # TODO MASSIVE HACK; mokC completely overlaps hokC in same reading frame, so matrix row is redundant
            if gene_to_count.locus_tag == 'b4412':
                continue

            gene_sequence = gene_to_count.sequence
            codons = [gene_sequence[i:i+3] for i in range(0, len(gene_sequence), 3)]
            for codon in codons:
                # str(codon) due to Biopython warning about older versions using object comparison for Seq
                codon_count_genes[str(codon)] += 1

        for codon, count in codon_count_matrix.items():
            if count != codon_count_genes[codon]:
                print(codon + ': codon counted directly from matrix and codons counted by gene do not match')
                validity = False

        # --- TEST: coding feature annotations only present within coding genes ---
        gene_cds_sub_matrix = self.extract(row_labels=['gene_CDS'], base_name=True)
        gene_cds_collapse = gene_cds_sub_matrix.sum(axis=0)
        coding_feat_sub_matrix = self.extract(row_labels=coding_related_features, base_name=True)
        coding_feat_collapse = coding_feat_sub_matrix.sum(axis=0)

        # if there are ANY coding features outside the gene_CDS range, then the sum of the collapsed matrix will have
        # a sum more than 1; NOTE: this works because coding_related_features has gene_CDS in it, a perfect mask adding
        # to 2 for all gene_CDS locations
        coding_feat_sum_vec = gene_cds_collapse + coding_feat_collapse
        if 1 in coding_feat_sum_vec:
            print('Coding sub-feature found outside range of CDS gene')
            validity = False

        # should be same or more number of stop codons than number of CDS genes
        stop_codons = codon_table.stop_codons
        stop_codon_sub_mat = self.extract(row_labels=['codon_' + sc for sc in stop_codons], base_name=True)
        stop_codon_sum = stop_codon_sub_mat.sum()
        if stop_codon_sum < len(genes_to_count):
            print('Fewer stop codons found than CDS genes')
            validity = False

        return validity

    def extract(
                self,
                row_labels: List[str] = None,
                column_range: Tuple[int, int] = None,
                base_name: bool = False,
                category: str = None
            ) -> csc_matrix:
        """
        Extract a specific set of rows for a given column range from the matrix; by default, extracts nothing

        :param List[str] row_labels: a list of row labels to pull out of the bitome matrix
        :param Tuple[int, int] column_range: a tuple in the form (column_start, column_end) for a position-based slice
        :param bool base_name: indicates if the row labels provided are base names (i.e. 'gene')
        :param str category: indicates the category of rows that should be considered; ignored unless row_labels is None
        :return csc_matrix matrix_window: the extracted matrix window requested
        """

        if row_labels is not None:
            all_row_indices = []
            if base_name:
                for row_base in row_labels:
                    all_row_indices += [i for i, label in enumerate(self.matrix_row_labels) if row_base in label]
            else:
                all_row_indices += [i for i, label in enumerate(self.matrix_row_labels) if label in row_labels]
        elif category is not None:
            all_row_indices = [
                i for i, row_category in
                enumerate(self.matrix_row_categories) if category == row_category
            ]
        else:
            all_row_indices = list(range(self.matrix.shape[0]))

        if column_range is not None:
            column_window = self.matrix[:, column_range[0]:column_range[1]]
            column_window_csr = column_window.tocsr()
            return column_window_csr[all_row_indices]
        else:
            return self.matrix.tocsr()[all_row_indices, :].tocsc()

    def save(self, file_name: str = 'bitome', dir_name: str = ''):
        """
        Dump this bitome instance to a pickle file for later loading; to reload a bitome from file, instantiate
        a new Bitome object that points at the file name where the pickle dump was created

        :param str file_name: the name of the dump file for this bitome object; defaults to 'bitome'
        :param str dir_name: the name of the directory of the dump file for this bitome object; defaults to ''
        """
        full_path = Path(dir_name, file_name + '.pkl')
        sys.setrecursionlimit(6000)
        with open(full_path, 'wb') as output_file:
            pickle.dump(self, output_file, pickle.HIGHEST_PROTOCOL)
        sys.setrecursionlimit(3000)

    def save_matrix(self, file_format: str = 'npz', transpose: bool = True, column_reduce: bool = True):
        """
        Saves a dump of the bitome matrix within this Bitome object; compatible with SciPy .npz and MatLab .mat formats

        :param str file_format: the format in which to save the matrix; either npz (default) or mat
        :param bool transpose: indicates if the bitome matrix should be transposed before saving; current form is
        :param bool column_reduce: indicates if duplicate columns should be found and eliminated
        feature rows x position columns (i.e. n_rows <<< n_cols)
        """

        if column_reduce:
            compressed_matrix = self._column_reduce_matrix()
        else:
            compressed_matrix = self.matrix

        if transpose:
            matrix_to_save = compressed_matrix.T
        else:
            matrix_to_save = compressed_matrix

        # REQUIRED for scipy.sparse.linalg.svds to work (or 'd')
        matrix_to_save = matrix_to_save.astype('f')

        num_rows = self.matrix.shape[0]
        if column_reduce:
            full_file_name = f'bitome_matrix_{num_rows}_col_red'
        else:
            full_file_name = f'bitome_matrix_{num_rows}'

        if file_format == 'npz':
            save_npz('matrix_data/' + full_file_name + '.npz', matrix_to_save, compressed=True)
        elif file_format == 'mat':
            savemat('matrix_data/' + full_file_name, {'bitome': matrix_to_save}, do_compression=True)
        else:
            print('Unrecognized file format ' + file_format)

    def _column_reduce_matrix(self, include_base_pairs=True):
        """
        Compress the bitome matrix by eliminating duplicate columns that are identical to other columns

        :param bool include_base_pairs: indicate if the base-pair encoding rows should be included
        :return:
        """

        col_sums_list = self.nt_byte_sizes.tolist()[0]
        col_index_sum_tups_sorted = sorted(enumerate(col_sums_list), key=lambda tup: tup[1])
        col_index_tup_groups = []
        unique_col_sums = []
        for unique_col_sum, col_index_tups in itertools.groupby(col_index_sum_tups_sorted, lambda tup: tup[1]):
            if unique_col_sum > 0:
                col_index_tup_groups.append(list(col_index_tups))
                unique_col_sums.append(unique_col_sum)

        # for each group of column sizes, we want to extract the matrix at those values; then delete duplicate
        # columns and return that matrix; do so by checking out the non-zero data indices
        compressed_matrix = coo_matrix(([], ([], [])), shape=(self.matrix.shape[0], 0))
        for col_index_tuples in col_index_tup_groups:
            group_col_indices = [tup[0] for tup in col_index_tuples]

            if include_base_pairs:
                col_size_matrix = self.matrix[:, group_col_indices]
            else:
                col_size_matrix = self.matrix[8:, group_col_indices]

            non_zero_elements = col_size_matrix.nonzero()
            non_zero_tuples = zip(non_zero_elements[0], non_zero_elements[1])
            sorted_non_zero = sorted(non_zero_tuples, key=lambda tup: tup[1])
            row_index_tups = []
            column_indices = []
            for column_index, index_tups_for_col in itertools.groupby(sorted_non_zero, lambda tup: tup[1]):
                row_index_tups.append(tuple([tup[0] for tup in index_tups_for_col]))
                column_indices.append(column_index)

            unique_row_index_tups = set(row_index_tups)
            new_matrix_indices = []
            for column_index, row_index_tup in enumerate(unique_row_index_tups):
                new_matrix_indices += [(row_index, column_index) for row_index in row_index_tup]
            new_matrix_rows = [tup[0] for tup in new_matrix_indices]
            new_matrix_cols = [tup[1] for tup in new_matrix_indices]
            new_matrix = coo_matrix(
                (np.ones(len(new_matrix_indices)), (new_matrix_rows, new_matrix_cols)),
                shape=(col_size_matrix.shape[0], len(unique_row_index_tups)), dtype='uint8'
            )
            compressed_matrix = hstack([compressed_matrix, new_matrix])

        return compressed_matrix.tocsr()

    @property
    def coding_genes(self):
        # only compare with TUs and proteins if K-12 MG1655 GenBank ID
        if self.genbank_id == 'NC_000913.3':
            return [
                gene for gene in self.genes
                if gene.gene_type == 'CDS' and gene.transcription_units and gene.protein is not None
            ]
        else:
            return [gene for gene in self.genes if gene.gene_type == 'CDS']

    @property
    def included_genes(self):
        non_cds_genes = [gene for gene in self.genes if gene.gene_type != 'CDS']
        return non_cds_genes + self.coding_genes

    @property
    def included_proteins(self):
        return [coding_gene.protein for coding_gene in self.coding_genes]

    @property
    def feature_categories(self):
        if self.matrix is None:
            return None
        elif len(self._feature_categories) > 0:
            return self._feature_categories
        else:
            self._feature_categories = set(self.matrix_row_categories)
            return self._feature_categories

    @property
    def sparsity(self):
        if self.matrix is None:
            return None
        else:
            self._sparsity = (1 - self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1]))
            return self._sparsity

    @property
    def rank(self):
        if self._rank is not None:
            return self._rank
        elif self.matrix is None:
            return None
        else:
            print('This will take a few minutes to calculate!')
            # the eps parameter value here was empirically determined to be the order of magnitude at which estimate
            # stabilizes; very roughly determined though, consider allowing user to set their own epsilon
            self._rank = estimate_rank(aslinearoperator(self.matrix.T.astype(float)), 0.0000001)
            return self._rank

    @property
    def nt_byte_sizes(self):
        if self.matrix is None or isinstance(self._nt_byte_sizes, np.matrix):
            return self._nt_byte_sizes
        else:
            self._nt_byte_sizes = self.matrix.sum(axis=0)
            return self._nt_byte_sizes

    @property
    def nt_byte_size_counts(self):
        if self.matrix is None or isinstance(self._nt_byte_size_counts, dict):
            return self._nt_byte_size_counts
        else:
            byte_sizes = self.nt_byte_sizes
            self._nt_byte_size_counts = Counter(byte_sizes.tolist()[0])
            return self._nt_byte_size_counts

    def get_gene(self, name: str):
        """
        Return a particular gene by name
        :param str name: the name of the feature to extract
        :return Gene gene: the Gene object with the given name (or None if not found)
        """
        return_gene = None
        for gene in self.genes:
            if gene.name == name:
                return_gene = gene
                break
        return return_gene
