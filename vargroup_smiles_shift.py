
from rdkit import Chem
import re

'''
Ring vargroup is attribute of the whole ring, but not of a specific position in a ring.
This file is about moving vargroups to first ring digit in SMILES.
Example:
[R1]C1CC([vR2])CN(c2nc3c(s2)C(N)CC([vR1])([Me])C3)C1 =>
[R1]C1[vR2]CCCN(c2nc3[vR1]c(s2)C(N)CC([Me])C3)C1

The purpose of the code is to provide training targets for corresponding images.

This operation breaks SMILES syntax, and is actually loosing information,
so after it a SMILES could not be later restored, or converted to mol, or fgsmiles, etc.

Instead it could be translated to a series of specific variants,
one SMILES for one vargroup position in a ring. 
'''

digit_pattern = re.compile('\d+')
brackets_pattern = re.compile('\[\S+?\]')
vr_pattern = re.compile('\[v\S+?\]')

def get_r_positions(smiles):
    "Finding the position of vR-groups in SMILES"
    r_positions = [[i.start(), i.end()] for i in vr_pattern.finditer(smiles)]
    return r_positions

def get_symbol_by_idx(smiles: str, r_position: list) -> str:
    "A function for getting R-group character from index"
    symbol = smiles[r_position[0]:r_position[1]]
    return symbol

def get_forbidden_indices(smiles: str) -> list:
    '''
    A function that returns finds forbidden indices from groups like [CH3], [C2H5]
    to find the desired digit.
    Since the numbers in these groups interfere with us in the correct definition
    of the numbers related to the cycle
    '''
    indices_list = []
    for i in re.finditer(brackets_pattern, smiles):
        for idx in range(i.start()+1, i.end()):
            indices_list.append(idx)
    return indices_list

def get_cycle_indices(smiles: str, forbidden_indices: list) -> dict:
    "A function for finding in a molecule all digits denoting a cycle, and their indices"
    cycle_indices = {}
    for i in re.finditer(digit_pattern, smiles):
        #We discard the numbers included in groups from dict_of_smarts or R-group
        if i.start() in forbidden_indices:
            continue
        if (i.group() in cycle_indices):
            cycle_indices[i.group()].append(i.start())
        else:
            cycle_indices[i.group()] = [i.start()]

    #Several cycles can correspond to one digit:
    #{'1': [5, 59], '2': [17, 23, 28, 35], '3': [31, 56]}
    #
    #For repeating cycles, create a subindex:
    #{'1': [5, 59], '2': [17, 23], '3': [31, 56], '2.1': [28, 35]}

    duplicates_dict = {}
    for digit in cycle_indices.keys():
        #If the length of the list> 2, then create a subindex for the digit
        if len(cycle_indices[digit]) > 2:
            for idx in range(1, int(len(cycle_indices[digit])/2)):
                sub_idx = f'{digit}.{idx}'
                duplicates_dict[sub_idx] = cycle_indices[digit][2+2*(idx-1):2+2*idx]
            cycle_indices[digit] = cycle_indices[digit][:2]

    cycle_indices.update(duplicates_dict)
    return cycle_indices

def check_range_and_count_distance(cycle_indices: dict, r_position: tuple) -> dict:
    '''
    Check if the R-group is included in the cycle and calculate the distance
    from the first digit and the last digit
    '''

    for digit, idx in cycle_indices.items():
        cycle_indices[digit].append(r_position[0] in range(*idx))
        cycle_indices[digit].append(abs(r_position[0] - idx[0]))
        cycle_indices[digit].append(abs(r_position[0] - idx[1]))
    return cycle_indices

def choose_digit(cycle_indices: dict, r_position: tuple, smiles_len: int) -> dict:
    '''
    A function for selecting the desired digit
    '''
    min_dist_right = min([value[-1] for value in cycle_indices.values()])
    min_dist_left = min([value[-2] for value in cycle_indices.values()])
    is_shift = False

    #The case when the R-group at the beginning of the molecule
    if 0 in range(*r_position):
        for digit, value in cycle_indices.items():
            if (value[-2] == min_dist_left):
                result_digit = digit
                result_index = value[0]
                is_shift = True
                break

    #The case when the R-group at the end of the molecule
    elif (smiles_len-1) in range(*r_position):
        for digit, value in cycle_indices.items():
            if (value[-1] == min_dist_right):
                result_digit = digit
                result_index = value[0]
                break

    #The case when the R-group is inside the cycle
    elif any([value[-3] for value in cycle_indices.values()]):
        min_dist_left = min([value[-2] for value in cycle_indices.values() if (value[-3] == True)])
        for digit, value in cycle_indices.items():
            if (value[-2] == min_dist_left) & (value[-3] == True):
                result_digit = digit
                result_index = value[0]
                break
    return result_digit, result_index, is_shift

def edit_smiles_one_group(smiles: str, r_position: tuple) -> str:
    '''
    A function that shifts the R-group to the cycle digit

    [R1]SCCC(NC(=O)c1cccc([R1])c1)C(=O)NCC(F)(F)F => [R1]SCCC(NC(=O)c1[vR1]ccccc1)C(=O)NCC(F)(F)F
    [R1]c1ccc(CN2CC([CH3])(C3CC3)NCCC2[Me])c(Cl)c1 => c1[vR1]ccc(CN2CC([CH3])(C3CC3)NCCC2[Me])c(Cl)c1

    '''
    #1. Find the required number to replace
    forbidden_indices = get_forbidden_indices(smiles)
    cycle_indices = get_cycle_indices(smiles, forbidden_indices)
    cycle_indices_and_distance = check_range_and_count_distance(cycle_indices, r_position)
    chosen_digit, chosen_index, is_shift = choose_digit(cycle_indices_and_distance, r_position, len(smiles))

    #2. Modifying SMILES
    r_symbol = get_symbol_by_idx(smiles, r_position)

    if is_shift:
        chosen_index -= (len(r_symbol))

    edited_smiles = ''.join([smiles[:r_position[0]], smiles[r_position[1]:]])
    edited_smiles = ''.join([edited_smiles[:chosen_index+1], r_symbol, edited_smiles[chosen_index+1:]])
    edited_smiles = edited_smiles.replace('()', '')
    return edited_smiles

def vargroup_smiles_shift (smiles: str) -> str:
    '''
    A function for editing smiles with multiple R-groups
    '''
    sorted_r_positions = sorted(get_r_positions(smiles), reverse=True)
    for r_position in sorted_r_positions:
        smiles = edit_smiles_one_group(smiles, r_position)
    return smiles

if __name__ == '__main__':
    print(edit_smiles_all_group('[R1]C1CC([vR2])CN(c2nc3c(s2)C(N)CC([vR1])([Me])C3)C1'))
