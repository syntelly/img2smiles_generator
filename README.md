# img2smiles_generator
Img2SMILES generator

This code supports the article: Image2SMILES: Transformer-based Molecular Optical Recognition Engine  
https://chemrxiv.org/engage/chemrxiv/article-details/60c758c6469df4169bf45744

Main purpose of the code - to generate datasets of pairs "image - sequence" for chemical molecules.
```
from data import Dumper
Dumper("name.csv").dump()
```
will generate image cach at `name_dump` and a file `name_result.csv`, containing target sequences and their pathcodes  
`name_fails.csv` contains list of failed molecules and the reasons  
`name_grpcounter.lst` is a list of counted unique substituted groups   
`smiles_test.csv` is an example of input smiles file.

draw.py could be used as standalone drawer  
fgsmiles.py could be used as standalone SMILES <-> Mol <-> FG-SMILES converter  
See inline comments in those files to get usage ideas.

To run the code, you need basically:  
`rdkit, opencv, cairosvg`

Please note that only rdkit v2020.09 is supported!  
2020.03 will generate bad images, 2021.03 works fine but has differencies in svg output format, so vargroups will not be drawn. 

Your installation insctructions could be:
```
$ conda install -c conda-forge rdkit=2020.09
$ conda install -c fastai opencv-python-headless
$ conda install -c conda-forge cairosvg
```
