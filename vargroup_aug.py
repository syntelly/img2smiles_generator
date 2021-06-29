from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

'''
Explaining some used names:
'Candidate atom', or just 'atom' - the one we try to put in variable position
'Neighbor atom', 'neighbor' or 'neib' - the one already connected to candidate atom
'Ring neighbor' - atoms, connected to 'neighbor atom' in the same ring.
One of ring neibs is gonna be chosen as second connection to candidate atom
'''

def toss_coin (n=2):
    # returns 1 of n outcome if n is int
    # if n is float, it is treated as raw probability
    if type(n) is int:
        return np.random.randint(n)==0
    else:
        return np.random.rand() < n

def check_atom (a, repl_symbs):
    #  if atom has FG from given list, not in ring,
    # connected to a ring neighbor by single bond
    if a.GetPropsAsDict().get("_fgsmilesLabel") not in repl_symbs: return False
    if len(a.GetNeighbors()) != 1: return False
    if a.GetBonds()[0].GetBondType() != Chem.rdchem.BondType.SINGLE: return False
    if not a.GetNeighbors()[0].IsInRing(): return False
    return True

def get_ring_atoms (a):
    #  returns list of atoms from the ring containing `a`
    #  `a` is member of single ring only, otherwise return None
    m = a.GetOwningMol()
    ri = m.GetRingInfo()
    all_rings = ri.AtomRings()
    aid = a.GetIdx()
    atom_rings = []
    for rids in all_rings:
        if aid in rids:
            atom_rings.append(rids)
    if len(atom_rings) != 1:
        return None
    else:
        return atom_rings[0]

def get_bond_idx (m, a1, a2):
    return m.GetBondBetweenAtoms(a1.GetIdx(), a2.GetIdx()).GetIdx()

def check_neib (n, a):
    #  if `n` is good neighbor for candidate `a`
    #  `n` must be carbon, only two ring neighbs (meaning not at rings crossing),
    # these two ring neibs must be carbons with no chirality
    m = a.GetOwningMol()
    if n.GetSymbol() != "C": return False
    other_neibs = [nn for nn in n.GetNeighbors() if nn.GetIdx() != a.GetIdx()]
    if len(other_neibs) != 2: return False # only 2 ring neibs
    for on in other_neibs:
        if on.GetSymbol() != "C":
            return False
        if on.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            return False
    return True

def check_ring (ring_ids, m):
    #  if the ring given bt list of atom ids is valid for variate position subst
    if len(ring_ids) not in [5,6,7]: # ring must be of size 5,6 or 7
        return False
    if len(ring_ids) == 5: # taking small rings with a 20% chance
        if not toss_coin(0.2):
            return False
    # gathering amounts of non-carbon atoms and external branches for each ring atom
    # one branch is the variated position atom itself
    n_ext_branches = 0
    n_non_carbon = 0
    for i in ring_ids:
        a = m.GetAtomWithIdx(i)
        neibs = a.GetNeighbors()
        n_ext_branches += max(len(neibs)-2, 0)
        if a.GetSymbol() != "C":
            n_non_carbon += 1
    if n_ext_branches > 3 and len(ring_ids) >= 6:
        return False # only 3 branches allowed for large rings
    if n_ext_branches > 2 and len(ring_ids) == 5:
        return False # only 2 branches allowed for small rings
    if n_non_carbon > 1: # only one heteroatom allowed, and with small chance
        return False
    elif n_non_carbon == 1:
        if not toss_coin(0.1): # 10% chance to pass with heterocycle
            return False
    if not a.GetIsAromatic(): # 20% chance to pass with non-aromatic ring
        if not toss_coin(0.2):
            return False
    return True

def vargroup_aug (mol, repl_symbs=['R', "R'", 'R"', 'R1', 'R2', 'R3', 'X', 'Y', 'Me', 'Et']):
    # Tries to put one or two atoms in `mol` to variated position
    # Allowed atom labels are taken from `repl_symbs` (generaly most common R-groups)
    # First need to define if there are such atoms in given molecule,
    # and also if the ring configuration is allowed to make the operation
    repl_symbs = [f'[{symb}]' for symb in repl_symbs] 
    candidate_atoms = [a for a in mol.GetAtoms() if check_atom(a, repl_symbs)]
    if len(candidate_atoms) == 0: # if no good positions found, return untouched mol
        return mol,[]

    # maximal two successful operations are allowed
    rwm = Chem.RWMol(mol)
    rbonds = []
    used_rings = []
    counter = 0
    counter_limit = 2
    aids = []
    np.random.shuffle(candidate_atoms)
    for a in candidate_atoms:
        # checking atom neighbor and the ring containing the neib
        n = a.GetNeighbors()[0]
        if not check_neib(n,a):
            continue

        ring_ids = get_ring_atoms(n)
        if ring_ids is None:
            continue

        if not check_ring(ring_ids, mol):
            continue

        # a ring can be used only once 
        if tuple(ring_ids) in used_rings:
            continue

        # taking ring neighbors of the candidate neighbor (there must be exactly two of them)
        ring_neibs = [rn for rn in n.GetNeighbors()
                      if (rn.GetSymbol() == 'C' and len(rn.GetNeighbors()) == 2)]

        if len(ring_neibs) != 2:
            continue

        # all checks are passed here, registering used ring
        used_rings.append(tuple(ring_ids))
        counter += 1
        
        # choosing one of two ring neibs, and add a new bond from it to candidate atom.
        # the new bond is written as ionic, otherwise there are valence issues in aromatic rings
        rn = np.random.choice(ring_neibs)
        rwm.AddBond(a.GetIdx(), rn.GetIdx(), Chem.rdchem.BondType.IONIC)
        
        b1id,b2id = get_bond_idx(rwm, a, rn), get_bond_idx(rwm, a, n)
        rbonds.append([b1id, b2id])
        aids.append(a.GetIdx())

        # also add 'v'prefix for candidate fgsmiles label and 'varbond' prop to delete later at mol2smiles call
        rwm.GetAtomWithIdx(a.GetIdx()).SetProp(
            "_fgsmilesLabel", '[v'+a.GetProp("_fgsmilesLabel")[1:])
        
        rwm.GetBondWithIdx(b1id).SetProp("_varbond", 'true')
        
    if len(rbonds) == 0:
        return mol,[]
    else:
        return Chem.Mol(rwm), rbonds

from fgsmiles import Mol2FGSmiles
