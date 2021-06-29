from tqdm import tqdm
from rdkit import Chem
import numpy as np
import os
import cv2
import pandas as pd
import csv

import hashlib
import binascii
from collections import Counter


def gethash (name):
    md5 = hashlib.md5(name.encode()).hexdigest()
    crc32 = str(hex(binascii.crc32(name.encode())))[2:]
    return md5+crc32

def decolorize (img):
    if not is_blackwhite(img):
        if np.random.rand() < 0.5:
            swapcolors = np.random.permutation([0,1,2])
            img = img[:,:,swapcolors]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


from draw import draw_mol_aug, is_blackwhite
from fgsmiles import Mol2FGMol, Mol2FGSmiles, get_fgsmiles_groups
from vargroup_aug import vargroup_aug
from vargroup_smiles_shift import vargroup_smiles_shift

class Dumper ():
    def __init__ (self, src_file, augment=True):
        self.src_smiles = np.array(pd.read_csv(src_file)).ravel()
        self.result = []
        self.fails =[]
        self.name = os.path.splitext(os.path.split(src_file)[-1])[0]
        self.dump_dir = self.name+"_dump"
        self.grpcounter = Counter()
        
    def draw_smiles_ (self, sm, replace_prob, vargroup_prob):
        if not np.random.rand() < replace_prob:
            img = draw_mol_aug(sm)
            return img, sm


        try:
            mol = Chem.MolFromSmiles(sm)
            fgmol = Mol2FGMol(mol, replace_all=False, do_generic=True)
            fgsm = Mol2FGSmiles(fgmol)
            
        except:
            print('failed to make FGMol')
            img = draw_mol_aug(sm)
            return img, sm

        try:
            if not(np.random.rand() < vargroup_prob): raise Exception()
            var_fgmol, varbonds = vargroup_aug(fgmol)
            if len(varbonds) == 0: raise Exception()
            var_fgsm = Mol2FGSmiles(var_fgmol)
            
            var_fgsm = vargroup_smiles_shift(var_fgsm)
            img = draw_mol_aug(var_fgmol, svg=True, varbonds=varbonds)
            return img, var_fgsm
        except Exception as e:
            if str(e) != '':
                print(e)
                print(var_fgsm)
            
            ez_flag = '/' in sm or '\\' in sm
            img = draw_mol_aug(fgmol, no_coordgen=ez_flag)
            return img, fgsm

    def draw_smiles (self, sm, replace_prob, vargroup_prob):
        img, tgt = self.draw_smiles_(sm, replace_prob, vargroup_prob)
        self.grpcounter.update(get_fgsmiles_groups(tgt, allow_generic=True))
        img = decolorize(img)
        return img, tgt
        
    def make_path (self, sm):
        name = gethash(sm)
        dir_l1 = os.path.join(self.dump_dir, name[0])
        dir_l2 = os.path.join(dir_l1, name[1])
        dir_l3 = os.path.join(dir_l2, name[2])
        img_path = os.path.join(dir_l3, name+".png")
        if not os.path.exists(dir_l3):
            try:
                os.makedirs(dir_l3)
            except:
                pass
        return img_path 
        
    def dump (self, nprocs=1):
        if not os.path.exists(self.dump_dir):
            os.mkdir(self.dump_dir)

    def dump_sample (self, idx):
        sm = str(self.src_smiles[idx])
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            self.fails.append([idx, sm, 'invalid smiles'])

        weights = [10,10,10,9,9,8,7,6,5,4,3,3,2,2,2,1]
        n_atoms = min(mol.GetNumAtoms(), len(weights)-1)
        n_samples = weights[n_atoms]

        def dump_fn (ntry, replace_prob, vargroup_prob):
            try:
                img, tgt_sm = self.draw_smiles(sm,  replace_prob, vargroup_prob)
                dump_path = self.make_path(tgt_sm+str(ntry))
                cv2.imwrite(dump_path, img)
                path_code =  os.path.splitext(os.path.split(dump_path)[1])[0]
                self.result.append([tgt_sm, path_code])
            except Exception as e:
                print(f'Failed to draw: {e}')
                self.fails.append([idx, sm, str(e)])
                
        if n_samples == 1:
            dump_fn(ntry=0, replace_prob=0.9, vargroup_prob=0.9)
        else:
            dump_fn(ntry=0, replace_prob=0, vargroup_prob=0)
            for i in range(n_samples-1):
                dump_fn(ntry=1+i, replace_prob=1, vargroup_prob=0.9)
        
    def dump (self):
        for idx in tqdm(range(len(self.src_smiles))):
            self.dump_sample(idx)

        pd.DataFrame(self.result).to_csv(
            self.name+"_result.csv",
            header=["smiles", "pathcode"],
            index=False,
            quoting=csv.QUOTE_NONE)

        with open(self.name+"_groupstat.lst", 'w') as fh:
            for grp,count in self.grpcounter.most_common():
                fh.write(f"{grp} : {count}\n")
        
        if len(self.fails) > 0:
            pd.DataFrame(self.fails).to_csv(
                self.name+"_fails.csv",
                header=["src_idx", "smiles", "error"],
                index=False,
                escapechar="\\",
                quoting=csv.QUOTE_NONE)

D = Dumper("smiles_tiny.csv")
D.dump()

