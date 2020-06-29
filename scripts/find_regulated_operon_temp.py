import numpy as np

from bitome.core import Bitome

test_bitome = Bitome.init_from_file('matrix_data/bitome.pkl')
operons = test_bitome.operons
op_regulon_counts = []
for operon in operons:
    tus = operon.transcription_units
    proms = [tu.promoter for tu in tus if tu.promoter is not None]
    regs = []
    for prom in proms:
        regs += prom.regulons
    reg_names = [reg.name for reg in regs]
    unique_reg_count = len(set(reg_names))
    op_regulon_counts.append(unique_reg_count)

max_locs = np.where(np.array(op_regulon_counts) == max(op_regulon_counts))[0]
for max_loc in max_locs:
    print(operons[max_loc].name)

