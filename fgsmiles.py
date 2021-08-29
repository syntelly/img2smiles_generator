import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import re

'''
Standalone FG-SMILES implementation.
FG-SMILES is a SMILES modification, allowing to insert functional groups shortcuts into SMILES as pseudo-atoms.
RDKit v 2020.09 and later allows to implement FG-SMILES engine quite straitforward.

Key functions are `Mol2FGMol` and `Mol2FGSmiles`.
Mol2FGMol do search and replace functional groups from _groups_ list.
Each found group is in molecule replaced with a pseudo-atom [U:n], where n is number of replacement, written as atom-mapping.
Drawing label is written into mol object as `_displayLabel` property, which is supported by latest rdkit.
By default MOl2FGMol has deterministic behaviour and replaces everything found, from larger groups to smaller ones.

Mol2FGSMiles generates FG-SMILES string out of modified mol. It uses information, saved in modified mol object,
to substitute corresponding [U:n] records to actual functional group labels.
'''

################################################################
### utils

# Each group may have more than one display label.
# This function selects one of several labels, according to their probability
# label_dict is {'lb' : prob}
def select_label (label_dict):
    labels, probs = [],[]
    for lbl,prob in label_dict.items():
        labels.append(lbl)
        probs.append(prob)

    lbl = np.random.choice(labels, p=np.array(probs))
    return f'[{lbl}]', gen_draw_label(lbl), gen_draw_label(mirror_label(lbl))

# generates display label string for a functional group label
# sending digits other stuff to subscripts or superscripts
def gen_draw_label (label):
    tokens = "<sub>", "</sub>"

    # sometimes i, s, t prefixes wre displayed as subscript or superscript
    if label in ['iPr', 'iBu', 'sBu', 'tBu']:
        if np.random.rand() < 0.05:
            if np.random.rand() < 0.5:
                tokens = "<sup>", "</sup>"
            # adding whitespace at front, otherwise RDKit segfaults
            # probably a drawing label cannot start from '<' char in current rdkit 
            return ' '+tokens[0]+label[0]+tokens[1]+' '+label[1:]

    # 25% chance to send ' and " upper
    if label in ["R'", 'R"']:
        if np.random.rand() < 0.25:
            tokens = "<sup>", "</sup>"
    # 10% chance to draw 1 2 or 3 as superscript
    elif label in ['R1', 'R2', 'R3']:
        if np.random.rand() < 0.1:
            tokens = "<sup>", "</sup>"
    chars = []
    # wrap these chars (mostly digits) with sub/sup
    for ch in label:
        if ch in "0123456789":
            chars.append(tokens[0]+ch+tokens[1])
        else:
            chars.append(ch)
    return "".join(chars)


################################################################
### search and replace

'''
If `select_all` is True, find every single match from _groups_.
Return reversed list, so that elder groups go before smaller ones.

If `select_all` is False, shuffle found matches list and select groups
by probability counted by `freq2prob`.

'''

def search_groups (mol, groups, select_all):
    selected = []
    for grp in groups:
        tmpl, _,_, freq,_ = grp
        n_matches = len(mol.GetSubstructMatches(tmpl))
        for i in range(n_matches):
            if select_all:
                prob = 1
            else:
                prob = freq2prob(freq)

            if np.random.rand() < prob:
                selected.append(grp)

    if not select_all:
        np.random.shuffle(selected)
    else:
        selected = selected[::-1]
    
    return selected

'''
Replacing given groups one by one with a pseudo-atom [U:n], n is incremental counter of replaces.
Replaced atom gets props: _displayLabel and _displayLabelW - supported by rdkit and used in drawing
_fgsmilesLabel prop is the string which gonna be substituted while generating FG-SMILES string.
Drawing label and fgsmiles label are not the same!
_substLabel is the target for fgsmiles substring replacement: _substLabel => _fgsmilesLabel

With these labels you do not need any supporting data to generate FG-SMILES out of the molecule

`do_generic` option allows to substitute `Me` group with a random generic group. Generic groups are those
which do not represent any real structure, but they are often used in practice. R-groups, GP, Pol, CxHy, etc
`Me` groups are found very often (>80% frequency), meaning there are enough methyls,
so 90% of found Me are replaced with generic groups  
 
'''

def replace_groups (mol, groups, do_generic):
    counter = 1
    def make_replacement (labelset):
        smiles_label, draw_label, draw_label_mirror = labelset
        nonlocal counter
        subst_label = f'[U+6:{counter}]'
        subst_tmpl = Chem.MolFromSmarts(subst_label)
        subst_tmpl.GetAtomWithIdx(0).SetProp("_displayLabel", draw_label)
        subst_tmpl.GetAtomWithIdx(0).SetProp("_displayLabelW", draw_label_mirror)
        subst_tmpl.GetAtomWithIdx(0).SetProp("_fgsmilesLabel", smiles_label)
        subst_tmpl.GetAtomWithIdx(0).SetProp("_substLabel", subst_label)
        counter += 1
        return subst_tmpl, subst_label
    
    for tmpl,_,labels,_,_ in groups:
        labelset = select_label(labels)
        _, draw_label, _ = labelset

        if draw_label == 'Me' and do_generic:
            if np.random.rand() < 0.9:
                labelset = select_label( {random_generic_group(): 1} )

        subst_tmpl, subst_label = make_replacement(labelset)
        mol = AllChem.ReplaceSubstructs(mol, tmpl, subst_tmpl, replaceAll=False)[0]

    return mol

################################################################
### toplevel functions

def Mol2FGMol (mol, replace_all=True, do_generic=False):
    use_groups = search_groups(mol, _groups_, replace_all)
    return replace_groups(mol, use_groups, do_generic)

def Mol2FGSmiles (mol):
    delbonds = []
    for b in mol.GetBonds():
        if b.HasProp("_varbond"):
            delbonds.append([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
    if len(delbonds) > 0:
        mol = Chem.RWMol(mol)
        for a1id, a2id in delbonds:
            mol.RemoveBond(a1id, a2id)
    
    sm = Chem.MolToSmiles(mol, canonical=False)        
    
    for a in mol.GetAtoms():
        props = a.GetPropsAsDict()
        if props.get("_displayLabel") is not None:
            frm = props.get("_substLabel")
            to = props.get("_fgsmilesLabel")
            if frm is None or to is None:
                raise Exception('Malformed FGMol #1')
            
            if frm not in sm:
                raise Exception('Malformed FGMol #2')

            sm = sm.replace(frm, to)
    
    return sm

group_template = re.compile('\[\S+?\]')
dummy_templates = [re.compile('\[C\d+H\d+\]'), re.compile('\[OC\d+H\d+\]')]

def bracket (grp):
    return '['+grp+']'

# this function converts FG-Smiles string to a valid Smiles string. Generic groups are replaced to `*`

def FGSmiles2Smiles (fgsmiles):
    for fg in get_fgsmiles_groups(fgsmiles, allow_generic=True):
        if fg in _unpack_list_:
            fgsmiles = fgsmiles.replace(bracket(fg), _unpack_dict_[fg])
        
        if fg in _generic_names_:
            fgsmiles = fgsmiles.replace(bracket(fg), '*')
        
        for tmpl in dummy_templates:
            fgsmiles = re.sub(tmpl, '*', fgsmiles)
            
    mol = Chem.MolFromSmiles(fgsmiles)
    if mol is None:
        raise Exception(f'Cannot convert {fgsmiles} into rdkit object')
    
    smiles = Chem.MolToSmiles(mol)
    return smiles

# find all groups from our list in input string

def get_fgsmiles_groups (fgsmiles, allow_generic=False):
    def _check (grp):
        if allow_generic:
            return grp in _unpack_list_ or \
                (grp[0] == 'v' and grp[1:] in _unpack_list_) or \
                grp in _generic_names_ or \
                (grp[0] == 'v' and grp[1:] in _generic_names_) or \
                max([len(tmpl.findall(f'[{grp}]')) > 0 for tmpl in dummy_templates]) > 0
            
        return grp in _unpack_list_
        
    groups = re.findall(group_template, fgsmiles)
    groups = [grp[1:-1] for grp in groups] # removing brackets [ ]
    groups = [grp for grp in groups if _check(grp)]
    return groups

# read fgsmiles to rdkit Mol object. It can be drawn and passed to Mol2FGSmiles

def MolFromFGSmiles (fgsmiles):
    groups = get_fgsmiles_groups(fgsmiles, allow_generic=True)

    # replace each group entrance with incremental [U+3:n] replacement
    # store replace pairs to a dict
    subst_dict = {}
    smiles = fgsmiles
    counter = 0
    for grp in groups:
        while bracket(grp) in smiles:
            counter += 1
            subst_str = f'U+3:{counter}'
            smiles = smiles.replace(bracket(grp), bracket(subst_str), 1)
            subst_dict[subst_str] = grp

    # after replacing to U+3:n the string becomes a valid SMILES
    # can read it with RDKit
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    # for each U+3:n atom set label properties, same as in Mol2FGMol
    for a in mol.GetAtoms():
        symbol = a.GetSmarts()
        if "[U+3:" in symbol:
            grp = subst_dict[symbol[1:-1]]
            a.SetProp("_displayLabel", grp)
            a.SetProp("_displayLabelW", mirror_label(grp))
            a.SetProp("_fgsmilesLabel", bracket(grp))
            a.SetProp("_substLabel", symbol)
            
    return mol

################################################################
### functional groups

# function group format: [SMARTS, {'label' : prob}, pubchem frequency, unpack template]
# after compilation:  [RDKit template, SMARTS, {'label' : prob}, pubchem frequency, unpack template]

_groups_ = [
    ['[F,Cl,Br,I]', {'X' : 0.4, 'Y': 0.25, 'Z': 0.15, 'Hal': 0.2}, 0.412, '*'],
    ['[CH3]', {'Me' : 0.9, 'CH3' : 0.1}, 0.811, 'C'],
    ['[CH2][CH3]', {'Et': 0.9, 'C2H5': 0.1}, 0.268, 'CC'],
    ['[CH2][CH2][CH3]', {'Pr': 0.9, 'C3H7': 0.1}, 0.083, 'CCC'],
    ['[CH1]([CH3])[CH3]', {'i-Pr': 0.9, 'iPr': 0.1}, 0.083, 'C(C)(C)'],
    ['[CH2][CH2][CH2][CH3]', {'Bu': 1}, 0.037, 'CCCC'],
    ['[CH2][CH1]([CH3])[CH3]', {'i-Bu': 0.9, 'iBu': 0.1}, 0.025, 'C(C(C)C)'],
    ['[CH1]([CH3])[CH2][CH3]', {'s-Bu': 0.9, 'sBu': 0.1}, 0.011, 'C(C)(CC)'],
    ['[CH0]([CH3])([CH3])[CH3]', {'t-Bu': 0.9, 'tBu': 0.1}, 0.043, 'C(C)(C)(C)'],
    ['O[CH3]', {'OMe': 0.9, 'OCH3': 0.1}, 0.182, 'O(C)'],
    ['O[CH2][CH3]', {'OEt': 0.9, 'OC2H5': 0.1}, 0.055, 'O(CC)'],
    ['O[CH2][CH2][CH3]', {'OPr': 0.9, 'OC3H7': 0.1}, 0.006, 'O(CCC)'],
    ['O[CH2][CH2][CH2][CH3]', {'OBu': 1}, 0.004, 'O(CCCC)'],
    ['[CH1](=O)', {'CHO': 1}, 0.011, 'C(=O)'],
    ['[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'Ph': 1.0}, 0.127, 'c9ccccc9'],
    ['O[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'OPh': 1.0}, 0.0064, 'O(c9ccccc9)'],
    ['[NH1][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'NHPh': 1.0}, 0.0045, 'N(c9ccccc9)'],
    ['[cH0]1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', {'Tol': 0.8, 'p-Tol': 0.2}, 0.023, 'c9(ccc(C)cc9)'],
    ['S(=O)(=O)[cH0]1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', {'Ts': 1}, 0.003, 'S(=O)(=O)(c9ccc(C)cc9)'],
    ['OS(=O)(=O)[cH0]1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', {'OTs': 1}, 0.0001, 'O(S(=O)(=O)c9ccc(C)cc9)'],
    ['[NH1]S(=O)(=O)[cH0]1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', {'NHTs': 1}, 0.001, 'N(S(=O)(=O)c9ccc(C)cc9)'],
    ['[CH0](=O)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'Bz': 1}, 0.008, 'C(=O)(c9ccccc9)'],
    ['[NH1][CH0](=O)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'NHBz': 1}, 0.002, 'N(C(=O)c9ccccc9)'],
    ['[CH0](=[O])[CH3]', {'Ac': 1}, 0.029, 'C(=O)(C)'],
    ['O[CH0](=[O])[CH3]', {'OAc': 1}, 0.005, 'O(C(=O)C)'],
    ['[NH1][CH0](=[O])[CH3]', {'NHAc': 1}, 0.009, 'N(C(=O)C)'],
    ['[CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'Bn': 1}, 0.041, 'C(c9ccccc9)'],
    ['O[CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'OBn': 1}, 0.009, 'O(Cc9ccccc9)'],
    ['[NH1][CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'NHBn': 1}, 0.004, 'N(Cc9ccccc9)'],
    ['[NH1][CH3]', {'NHMe': 1}, 0.322, 'N(C)'],
    ['[NH1][CH2][CH3]', {'NHEt': 1}, 0.399, 'N(CC)'],
    ['[NH1][CH2][CH2][CH3]', {'NHPr': 1}, 0.123, 'N(CCC)'],
    ['[NH1][CH2][CH2][CH2][CH3]', {'NHBu': 1}, 0.022, 'N(CCCC)'],
    ['[CH0](=O)O[CH0]([CH3])([CH3])[CH3]', {'Boc': 1}, 0.011, 'C(=O)(OC(C)(C)C)'],
    ['O[CH0](=O)O[CH0]([CH3])([CH3])[CH3]', {'OBoc': 1}, 0.0001, 'O(C(=O)OC(C)(C)C)'],
    ['[NH1][CH0](=O)O[CH0]([CH3])([CH3])[CH3]', {'NHBoc': 1}, 0.005, 'N(C(=O)OC(C)(C)C)'],
    ['[CH0](=O)O[CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'Cbz': 1}, 0.003, 'C(=O)(OCc9ccccc9)'],
    ['O[CH0](=O)O[CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'OCbz': 1}, 0.00001, 'O(C(=O)OCc9ccccc9)'],
    ['[NH1][CH0](=O)O[CH2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'NHCbz': 1}, 0.001, 'N(C(=O)OCc9ccccc9)'],
    ['[CH0](F)(F)F', {'CF3': 1}, 0.044, 'C(F)(F)(F)'],
    ['S(=O)(=O)C(F)(F)F', {'Tf': 1}, 0.0006, 'S(=O)(=O)(C(F)(F)F)'],
    ['OS(=O)(=O)C(F)(F)F', {'OTf': 1}, 0.0002, 'O(S(=O)(=O)C(F)(F)F)'],
    ['[CH0](=O)[CH0]([CH3])([CH3])[CH3]', {'Piv': 1}, 0.002, 'C(=O)(C(C)(C)C)'],
    ['O[CH0](=O)[CH0]([CH3])([CH3])[CH3]', {'OPiv': 1}, 0.0003, 'O(C(=O)C(C)(C)C)'],
    ['[CH1]=[CH2]', {'Vin': 0.5, 'Vi': 0.5}, 0.017, 'C=C'],
    ['[CH2][CH1]=[CH2]', {'All': 1}, 0.013, 'C(C=C)'],
    ['[SiH0]([CH3])([CH3])[CH3]', {'TMS': 0.5, 'SiMe3': 0.5}, 0.001, '[Si](C)(C)(C)'],
    ['O[SiH0]([CH3])([CH3])[CH3]', {'OTMS': 0.5, 'OSiMe3': 0.5}, 0.006, 'O([Si](C)(C)C)'],
    ['[SiH0]([CH3])([CH3])[CH0]([CH3])([CH3])[CH3]', {'TBS': 0.5, 'TBDMS': 0.5}, 0.002, '[Si]((C)(C)C(C)(C)C)'],
    ['O[SiH0]([CH3])([CH3])[CH0]([CH3])([CH3])[CH3]', {'OTBS': 0.5, 'OTBDMS': 0.5}, 0.002, 'O([Si](C)(C)C(C)(C)C)'],
    ['[CH1]1O[CH2][CH2][CH2][CH2]1', {'THP': 1}, 0.002, 'C9(OCCCC9)'],
    ['O[CH1]1O[CH2][CH2][CH2][CH2]1', {'OTHP': 1}, 0.0005, 'O(C9OCCCC9)'],
    ['[SiH0]([cH0]1[cH1][cH1][cH1][cH1][cH1]1)([cH0]1[cH1][cH1][cH1][cH1][cH1]1)[CH0]([CH3])([CH3])[CH3]', {'TBDPS': 1}, 0.0002, '[Si]((c9ccccc9)(c9ccccc9)C(C)(C)C)'],
    ['O[SiH0]([cH0]1[cH1][cH1][cH1][cH1][cH1]1)([cH0]1[cH1][cH1][cH1][cH1][cH1]1)[CH0]([CH3])([CH3])[CH3]', {'OTBDPS': 1}, 0.0002, 'O([Si](c9ccccc9)(c9ccccc9)C(C)(C)C)'],
    ['O[CH2]O[CH3]', {'OMOM': 1}, 0.0003, 'O(COC)'],
    ['[SiH0]([CH2][CH3])([CH2][CH3])[CH2][CH3]', {'TES': 0.5, 'SiEt3':0.5}, 0.0002, '[Si](CC)(CC)(CC)'],
    ['O[SiH0]([CH2][CH3])([CH2][CH3])[CH2][CH3]', {'OTES': 0.5, 'OSiEt3':0.5}, 0.0001, 'O([Si](CC)(CC)CC)'],
    ['[SiH0]([CH3])([CH3])[CH1]([CH3])[CH3]', {'IPDMS': 1}, 0.000001, '[Si]((C)(C)C(C)C)'],
    ['O[SiH0]([CH3])([CH3])[CH1]([CH3])[CH3]', {'OIPDMS': 1}, 0.00001, 'O([Si](C)(C)C(C)C)'],
    ['[SiH0]([CH2][CH3])([CH2][CH3])[CH1]([CH3])[CH3]', {'DEIPS': 1}, 0.00001, '[Si]((CC)(CC)C(C)C)'],
    ['O[SiH0]([CH2][CH3])([CH2][CH3])[CH1]([CH3])[CH3]', {'ODEIPS': 1}, 0.00001, 'O([Si](CC)(CC)C(C)C)'],
    ['[SiH0]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]', {'TIPS': 1}, 0.0001, '[Si]((C(C)(C))(C(C)(C))C(C)(C))'],
    ['O[SiH0]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])[CH1]([CH3])[CH3]', {'OTIPS': 1}, 0.0001, 'O([Si](C(C)(C))(C(C)(C))C(C)(C))'],
    ['[CH1]1O[SiH0]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])O[SiH0]([CH1]([CH3])[CH3])([CH1]([CH3])[CH3])O[CH1]1', {'TIPDS': 1}, 0.00001, '[Si]((C(C)C)(C(C)C)C(C)C)'],
    ['[CH0](=O)C(F)(F)F', {'TFA': 1}, 0.002, 'C(=O)(C(F)(F)F)'],
    ['O[CH0](=O)C(F)(F)F', {'OTFA': 1}, 0.0006, 'O(C(=O)C(F)(F)F)'],
    ['[CH0](=O)O[CH2][CH1]1[cH0]2[cH1][cH1][cH1][cH1][cH0]2-[cH0]2[cH1][cH1][cH1][cH1][cH0]21', {'Fmoc': 1}, 0.0003, 'C(=O)(OCC8c9ccccc9-c9ccccc98)'],
    ['O[CH0](=O)O[CH2][CH1]1[cH0]2[cH1][cH1][cH1][cH1][cH0]2-[cH0]2[cH1][cH1][cH1][cH1][cH0]21', {'OFmoc': 1}, 0.00001, 'O(C(=O)OCC8c9ccccc9-c9ccccc98)'],
    ['[NH1][CH0](=O)O[CH2][CH1]1[cH0]2[cH1][cH1][cH1][cH1][cH0]2-[cH0]2[cH1][cH1][cH1][cH1][cH0]21', {'NHFmoc': 1}, 0.0003, 'N(C(=O)OCC8c9ccccc9-c9ccccc98)'],
    ['[CH0](=O)O[CH2][CH1]=[CH2]', {'Alloc': 1}, 0.0006, 'C(=O)(OCC=C)'],
    ['O[CH0](=O)O[CH2][CH1]=[CH2]', {'OAlloc': 1}, 0.00001, 'O(C(=O)OCC=C)'],
    ['[NH1][CH0](=O)O[CH2][CH1]=[CH2]', {'NHAlloc': 1}, 0.0002, 'N(C(=O)OCC=C)'],
    ['[CH0](=O)O[CH2][CH0](Cl)(Cl)Cl', {'Troc': 1}, 0.0001, 'C(=O)(OCC(Cl)(Cl)Cl)'],
    ['O[CH0](=O)O[CH2][CH0](Cl)(Cl)Cl', {'OTroc': 1}, 0.00001, 'O(C(=O)OCC(Cl)(Cl)Cl)'],
    ['[NH1][CH0](=O)O[CH2][CH0](Cl)(Cl)Cl', {'NHTroc': 1}, 0.00001, 'N(C(=O)OCC(Cl)(Cl)Cl)'],
    ['[CH0](=O)O[CH2][CH2][SiH0]([CH3])([CH3])[CH3]', {'Teoc': 1}, 0.00001, 'C(=O)(OCC[Si](C)(C)C)'],
    ['O[CH0](=O)O[CH2][CH2][SiH0]([CH3])([CH3])[CH3]', {'OTeoc': 1}, 0.00001, 'O(C(=O)OCC[Si](C)(C)C)'],
    ['[NH1][CH0](=O)O[CH2][CH2][SiH0]([CH3])([CH3])[CH3]', {'NHTeoc': 1}, 0.00001, 'N(C(=O)OCC[Si](C)(C)C)'],
    ['[CH0]([cH0]1[cH1][cH1][cH1][cH1][cH1]1)([cH0]1[cH1][cH1][cH1][cH1][cH1]1)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'Tr': 1}, 0.08, 'C(c9ccccc9)(c9ccccc9)(c9ccccc9)'],
    ['O[CH0]([cH0]1[cH1][cH1][cH1][cH1][cH1]1)([cH0]1[cH1][cH1][cH1][cH1][cH1]1)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'OTr': 1}, 0.08, 'O(C(c9ccccc9)(c9ccccc9)c9ccccc9)'],
    ['[NH1][CH0]([cH0]1[cH1][cH1][cH1][cH1][cH1]1)([cH0]1[cH1][cH1][cH1][cH1][cH1]1)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'NHTr': 1}, 0.08, 'N(C(c9ccccc9)(c9ccccc9)c9ccccc9)'],
    ['[CH0](=S)[NH0]([CH3])([CH3])', {'DMTC': 1}, 0.0001, 'C(=S)(N(C)C)'],
    ['O[CH0](=S)[NH0]([CH3])([CH3])', {'ODMTC': 1}, 0.00001, 'O(C(=S)N(C)C)'],
    ['[BH0]1O[CH0]([CH3])([CH3])[CH0]([CH3])([CH3])O1', {'BPin': 1}, 0.0003, 'B9(OC(C)(C)C(C)(C)O9)'],
    ['[CH0](=O)[CH2][CH2][CH0](=O)[CH3]', {'Lev': 1}, 0.0001, 'C(=O)(CCC(=O)C)'],
    ['O[CH0](=O)[CH2][CH2][CH0](=O)[CH3]', {'OLev': 1}, 0.00001, 'O(C(=O)CCC(=O)C)'],
    ['[cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', {'PMP': 1}, 0.0213, 'c9(ccc(OC)cc9)'],
    ['O[cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', {'OPMP': 1}, 0.0014, 'O(c9ccc(OC)cc9)'],
    ['[CH2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', {'PMB': 1}, 0.0053, 'C(c9ccc(OC)cc9)'],
    ['O[CH2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', {'OPMB': 1}, 0.0003, 'O(Cc9ccc(OC)cc9)'],
    ['[Sv2][CH3]', {'SMe': 1}, 0.0251, 'S(C)'],
    ['[Sv2][CH2][CH3]', {'SEt': 1}, 0.0044, 'S(CC)'],
    ['[Sv2][cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'SPh': 1}, 0.0018, 'S(c9ccccc9)'],
    ['N=[N+]=[N-]', {'N3': 1}, 0.0012, 'N(=[N+]=[N-])'],
    ['[cH0]1n[cH0]2[cH1][cH1][cH1][cH1][cH0]2s1', {'Bt': 1}, 0.0036, 'c8(nc9ccccc9s8)'],
    ['[N+](=O)[O-]', {'NO2': 1}, 0.0432, '[N+](=O)([O-])'],
    ['O[CH0](=O)[CH1](O[CH3])[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'OMPA': 1}, 0.00001, 'O(C(=O)C(OC)c9ccccc9)'],
    ['[C]#[NH0]', {'CN': 1}, 0.5491, 'C(#N)'],
    ['[CH0](=O)[OH1]', {'CO2H': 0.5, 'COOH': 0.5}, 0.08, 'C(=O)(O)'],
    ['[CH0](=O)Cl', {'COCl': 1}, 0.001, 'C(=O)(Cl)'],
    ['[Sv6](=O)(=O)[CH3]', {'SO2Me': 1}, 0.0112, 'S(=O)(=O)(C)'],
    ['[Sv6](=O)(=O)[CH2][CH3]', {'SO2Et': 1}, 0.002, 'S(=O)(=O)(CC)'],
    ['[Sv6](=O)(=O)[cH0]1[cH1][cH1][cH1][cH1][cH1]1', {'SO2Ph': 1}, 0.0026, 'S(=O)(=O)(c9ccccc9)'],
    ['[Sv6](=O)(=O)[OH1]', {'SO3H': 1}, 0.0019, 'S(=O)(=O)(O)'],
    ['[cH0]1[cH0]([CH3])[cH1][cH0]([CH3])[cH1][cH0]1[CH3]', {'Mes': 0.8, 'Ms': 0.2}, 0.0022, 'c9(c(C)cc(C)cc9C)'],
    ['[BH0]([OH])[OH]', {'B(OH)2': 1}, 0.0005, 'B(O)(O)'],
    ['[CH0](=O)O[CH3]', {'CO2Me': 0.5, 'COOMe': 0.5}, 0.003, 'C(=O)(OC)'],
    ['[CH0](=O)O[CH2][CH3]', {'CO2Et': 0.5, 'COOEt': 0.5}, 0.0234, 'C(=O)(OCC)'],
    ['[NH1][OH1]', {'NHOH': 1}, 0.001, 'N(O)'],
    ['[NH1][NH2]', {'NHNH2': 1}, 0.0069, 'NN'],
    ['[N]([CH3])[CH3]', {'NMe2': 1}, 0.0302, 'N((C)C)'],
    ['[N]([CH2][CH3])[CH2][CH3]', {'NEt2': 1}, 0.0081, 'N((CC)CC)'],
    ['[CH1]1[CH2][CH2][CH2][CH2][CH2]1', {'Cy': 1}, 0.0123, 'C9CCCCC9'],
    ]

# compiled templates are really used instead of text versions. saving CPU time
def compile_templates (groups):
    for i,grp in enumerate(groups):
        groups[i] = [Chem.MolFromSmarts(grp[0])] + grp
    return groups

compile_templates(_groups_)

# empirical function to calculate replace probability from group frequency
# gives greater probs for less frequent groups, and almost 100% prob for most rare groups
# not less than 20% even for most frequent groups
def freq2prob (freq):
    return (1-freq**0.125)*0.8+0.2


################################################################
### generic groups

_generic_groups_ = {
    'R': 0.25, 'R1': 0.25, 'R2': 0.2, 'R3': 0.15, 'R4': 0.1, 'R5':0.1, 'R6': 0.1,
    'R7': 0.05, 'R8': 0.05, 'R9': 0.05, 'R10': 0.05, "R'": 0.2, 'R"': 0.1,
    'OR' : 0.025, "OR'" : 0.01, 'NHR' : 0.01, "NHR'" : 0.005, 
    'Ar': 0.05, 'PG': 0.01, 'Pol': 0.01, 'EWG' : 0.01, '#' : 0.005,
    'CxHy' : 0.1, 'OCxHy' : 0.05 }

_generic_probs_ = np.array(list(_generic_groups_.values()))
_generic_probs_ /= np.sum(_generic_probs_)
_generic_names_ = list(_generic_groups_.keys())

# just a random group grom _generic_groups_ list
# if CxHy or OCxHy were chosen, generate real numbers instead of x and y
# C3H7 and smaller already covered
def random_generic_group ():
    choice = np.random.choice(_generic_names_, p=_generic_probs_)
    if choice == 'CxHy':
        x = np.random.randint(10)+5 # 4-14
        y = x+np.random.randint(10)+3 # 2-12
        choice = f'C{x}H{y}'

    if choice == 'OCxHy':
        x = np.random.randint(5)+5 # 4-9
        y = x+np.random.randint(5)+3 # 2-7
        choice = f'OC{x}H{y}'
        
    return choice

################################################################
### unpack dict

_unpack_dict_ = {}

for _,_,labels,_,unpack_tmpl in _groups_:
    for lbl,_ in labels.items():
        _unpack_dict_[lbl] = unpack_tmpl

_unpack_list_ = list(_unpack_dict_.keys())

################################################################
### label mirroring stuff

# Instructions how to get _displayLabelW prop from _displayLabel

_mirror_groups_ = {'OMe': 'MeO',
                   'OCH3': 'H3CO',
                   'OEt': 'EtO',
                   'OC2H5': 'H5C2O',
                   'OPr': 'PrO',
                   'OC3H7': 'H7C3O',
                   'OBu': 'BuO',
                   'OPh': 'PhO',
                   'OTs': 'TsO',
                   'OAc': 'AcO',
                   'OBn': 'BnO',
                   'OBoc': 'BocO',
                   'OCbz': 'CbzO',
                   'OTf': 'TfO',
                   'OPiv': 'PivO',
                   'OTr': 'TrO',
                   'OTMS': 'TMSO',
                   'OSiMe3': 'Me3SiO',
                   'OTBS': 'TBSO',
                   'OTBDMS': 'TBDMSO',
                   'OTHP': 'THPO',
                   'OTBDPS': 'TBDPSO',
                   'OMOM': 'MOMO',
                   'OTES': 'TESO',
                   'OSiEt3': 'Et3SiO',
                   'OIPDMS': 'IPDMSO',
                   'ODEIPS': 'DEIPSO',
                   'OTIPS': 'TIPSO',
                   'OTFA': 'TFAO',
                   'OFmoc': 'FmocO',
                   'OAlloc': 'AllocO',
                   'OTroc': 'TrocO',
                   'OTeoc': 'TeocO',
                   'ODMTC': 'DMTCO',
                   'OLev': 'LevO',
                   'OPMP': 'PMPO',
                   'OPMB': 'PMBO',
                   'OMPA': 'MPAO',
                   'NO2': 'O2N',
                   'CHO': 'OHC',
                   'CO2H': 'HO2C',
                   'CO2Me': 'MeO2C',
                   'CO2Et': 'EtO2C',
                   'COOH': 'HOOC',
                   'COOMe': 'MeOOC',
                   'COOEt': 'EtOOC',
                   'COCl': 'ClOC',
                   'CH3': 'H3C',
                   'CF3': 'F3C',
                   'CN': 'NC',
                   'SMe': 'MeS',
                   'SEt': 'EtS',
                   'SPh': 'PhS',
                   'SO2Me': 'MeO2S',
                   'SO2Et': 'EtO2S',
                   'SO2Ph': 'PhO2S',
                   'SO3H': 'HO3S',
                   'NHOH': 'HOHN',
                   'NMe2': 'Me2N',
                   'NEt2': 'Et2N',
                   'NHNH2': 'H2NHN',
                   'NHTs': 'TsHN',
                   'NHBz': 'BzHN',
                   'NHAc': 'AcHN',
                   'NHBn': 'BnHN',
                   'NHMe': 'MeHN',
                   'NHEt': 'EtHN',
                   'NHPr': 'PrHN',
                   'NHBu': 'BuHN',
                   'NHPh': 'PhHN',
                   'NHBoc': 'BocHN',
                   'NHCbz': 'CbzHN',
                   'NHFmoc': 'FmocHN',
                   'NHAlloc': 'AllocHN',
                   'NHTroc': 'TrocHN',
                   'NHTeoc': 'TeocHN',
                   'NHTr': 'TrHN',
                   'B(OH)2': '(OH)2B',
                  }


def mirror_label (lbl):
    if lbl[0] == 'C':
        try:
            x,y = lbl[1:].split('H')
            x,y = int(x), int(y)
            return f'H{y}C{x}'
        except: pass
    
    if lbl[:2] == 'OC':
        try:
            x,y = lbl[2:].split('H')
            x,y = int(x), int(y)
            return f'H{y}C{x}O'
        except: pass
    
    return _mirror_groups_.get(lbl, lbl)
    
    
# simple self test

sm = 'CCCOC(=O)c1ccc(N2C[C@@H](C(=O)O[C@@H](CC)C(=O)c3ccccc3)CC2=O)cc1'
fgmol = Mol2FGMol(Chem.MolFromSmiles(sm))
fgsmiles = Mol2FGSmiles(fgmol)
MolFromFGSmiles(fgsmiles)
fgsmiles2 = Mol2FGSmiles(fgmol)
print(fgsmiles == fgsmiles2)
