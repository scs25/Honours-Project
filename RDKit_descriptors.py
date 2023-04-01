
import numpy as np
import pandas as pd
import os
import traceback

from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, rdmolops
from rdkit.ML.Descriptors import MoleculeDescriptors
import rdkit.Chem.Fragments as Fragments
from rdkit.Chem.rdchem import AtomValenceException

smiles = [] 
names = []

directory = os.fsencode('smi')

for file in os.listdir(directory): 
    filename = os.fsdecode(file) 
    if 'precat' in filename:
        if 'mer' not in filename:
            f = open('smi\\'+filename, "r") 
            contents = f.read().split() 
            names.append(contents[1].split('.')[0])
            smiles.append(contents[0]) 
        else:
            continue

molecules = [Chem.MolFromSmiles(i, sanitize=False) for i in smiles] 

descriptors = pd.DataFrame(index=names)

for i, mol in enumerate(molecules):
    mol.UpdatePropertyCache(strict=False)
    Chem.Kekulize(mol) # Converts Aromatic Rings to their Kekule Form
    Chem.SetAromaticity(mol) #  Identifies the Aromatic Rings and Ring Systems, sets the Aromatic Flag on Atoms and Bonds
    Chem.SetConjugation(mol) # Identifies which Bonds are Conjugated
    Chem.SetHybridization(mol) # Calculates the Hybridization State of each Atom
    molecules[i] = mol

desc_list = [n[0] for n in Descriptors._descList]

def compute_2Drdkit():
    rdkit_2d_desc = []
    physicochemical = [i for i in desc_list if not i.startswith('fr_')]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(physicochemical)
    header = physicochemical  

    rdkit_desc_sub = [calc.CalcDescriptors(mol) for mol in molecules]

    df = pd.DataFrame(rdkit_desc_sub, columns=header)  
    df.insert(loc=0, column='Names', value=names)  
    df.to_csv('Ligand Data/precat_descriptors_2.csv', index=False) 

compute_2Drdkit()