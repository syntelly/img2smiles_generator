# img2smiles_generator
Img2SMILES generator

Basically you need:
rdkit, opencv, cairosvg

Please note that only rdkit v2020.09 is supported!
2020.03 will generate bad images, 2021.03 works fine but has differencies in svg output format, so vargroups will not be drawn. 

Your installation insctructions could be:
```
$ conda install -c conda-forge rdkit=2020.09
$ conda install -c fastai opencv-python-headless
$ conda install -c conda-forge cairosvg
```
