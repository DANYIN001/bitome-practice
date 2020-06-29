# built-in modules
from pathlib import Path

# local modules
from bitome.core import Bitome

test_bitome = Bitome.init_from_file(Path('matrix_data', 'bitome.pkl'))
sequence = str(test_bitome.sequence)

keys_dict = {base: [sequence.find(base)] for base in ['A', 'C', 'G', 'T']}

index = 0
increment = 1
while True:
    if index % 100000 == 0:
        print(index)
    if not len(sequence) >= index+increment:
        break
    sub_str = sequence[index:index + increment]
    if sub_str in keys_dict:
        keys_dict[sub_str] = keys_dict[sub_str] + [index]
        increment += 1
    else:
        keys_dict[sub_str] = [index]
        index += increment
        increment = 1

codes = list(keys_dict.keys())
sorted_codes = sorted(codes, key=lambda code: len(code), reverse=True)
print(sorted_codes[:10])
