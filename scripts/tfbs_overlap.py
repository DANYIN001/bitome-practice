import itertools

from bitome.core import Bitome

test_bitome = Bitome.init_from_file('matrix_data/bitome.pkl')
tfs = test_bitome.transcription_factors
tfs_with_overlap = set()
for tf in tfs:
    for final_state, fs_binding_sites in tf.binding_sites.items():
        if fs_binding_sites:
            bs_locations = [bs.location for bs in fs_binding_sites]
            bs_loc_range_sets = [set(range(loc.start.position, loc.end.position)) for loc in bs_locations]
            for bs_loc_range_set_pair in itertools.combinations(bs_loc_range_sets, 2):
                bs_loc_intersection = set.intersection(*list(bs_loc_range_set_pair))
                if len(bs_loc_intersection) > 0:
                    tfs_with_overlap.add(tf)