
_USE_PUBCHEM_ = False # enable for production usage

'''
This file can be used as a standalone drawer. In that case disable vargroup-depending imports.
For that purpose this file does not depend on opencv, only PIL and numpy is used for image ops.
Anyway opencv is not used for image read-write operations, to avoid any unexpected RGB vs BGR issues
All images are returned in 3-channel mode, even if they are pure black&white
'''

from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import io
import PIL, PIL.Image, PIL.ImageChops
from PIL import ImageOps
from glob import glob
import os
import subprocess

# comment these for standalone usage
from svg_fix_vargroup import svg_fix_vargroup
from vargroup_aug import toss_coin

# os.system with hard timeout 
def system_call (command, timeout):
    return subprocess.check_output(command, shell=True, timeout=timeout)

fontdir = "fonts"
font_paths = glob(os.path.join(fontdir, "*.ttf"))

# for debug
def show (img):
    PIL.Image.fromarray(img).save("/tmp/show.png")
    os.system("feh /tmp/show.png")

from rdkit.Chem import rdDepictor, rdCoordGen
from cairosvg import surface
################################################################

'''
Rdkit drawer is implemented as recursive function and a toplevel wrapper. Probably not the clearest solution, but it works.
The problem is - you need to define canvas shape to draw a mol with rdkit. If the shape is too low, then rdkit will try
fit the mol into given shape and break proportions. If we set very big shape and then cut excessive whitespace, it will
take unnecessary time to draw - drawing time in rdkit depends on canvas area. We found that the following way takes optimal time.
First we guess shape according to number of heavy atoms: 150+n*15. If there are some black pixels in outer frame of 30px,
meaning the molecule touches borders, then we increase guessed shape 1.5 times and redraw. Sometimes (for long carbon chains)
it is also not enough, that is why we make up to 3 tries of redraw, increasing canvas shape 1.5 times each time.

Initially the drawer was implemented for rdkit 2020.03. Rdkit drawing API is unstable and different versions can be incompartible.
There were some bugs in 2020.03, and we moved to 2020.09 and later, where these bugs are fixed.
Unfortunately, in 2020.09 the CoordGen (alternative atom coodrinate generator) is broken: draws endbonds several times longer than middle bonds,
100% segfaults with EZ-stereo if mol was modified manually, and also some random segfaults.
Probably the long endbonds problem is connected to fixed length drawing, but we cannot use CoordGen in its current state.
All CoordGen functionality is left in place, but its actual call is commented.

'''

# recursive wrapper, giving max 3 tries if shape is not predefined
def draw_mol_rdkit (sm, shape=None, font_size=0.8, font_file=None, line_length=23, line_width=2,
              multiple_bond_offset=0.13, rotate=0, atom_padding=0.15,
              use_coordgen=False, svg=False, varbonds=[]):
    
    if shape is None:
        tries_left = 2
    else:
        shape = (shape,shape)
        tries_left = 0
    return draw_mol_rdkit_(sm, shape, font_size, font_file, line_length, line_width, multiple_bond_offset,
                    rotate, atom_padding, use_coordgen, svg, varbonds, tries_left)

# recursive calling of underlying drawer.
def draw_mol_rdkit_ (sm, shape, font_size, font_file, line_length, line_width,
              multiple_bond_offset, rotate, atom_padding,
              use_coordgen, svg, varbonds, tries_left):
    img = draw_mol_rdkit__(sm, shape, font_size, font_file, line_length, line_width, multiple_bond_offset,
                    rotate, atom_padding, use_coordgen, svg, varbonds)        

    # checking if outer frame contains black pixels
    # margin size is 5% of image width, but not less than 30 pixels
    H,W = img.shape[:2]
    inv_img = 255-img
    margin = max(30, int(W*0.05))
    inv_img[margin:-margin, margin:-margin] = 0
    if inv_img.sum() == 0 or tries_left <= 0:
        return img
    else:
        # if black pixels found, redraw with 1.5x shape
        shape = (int(W*1.5), int(H*1.5))
        return draw_mol_rdkit_(sm, shape, font_size, font_file, line_length, line_width,
                        multiple_bond_offset, rotate, atom_padding,
                        use_coordgen, svg, varbonds, tries_left-1)

# underlying drawer
def draw_mol_rdkit__ (sm, shape, font_size, font_file, line_length,
                      line_width, multiple_bond_offset,
                      rotate, atom_padding, use_coordgen, svg, varbonds):

    if type(sm) is str or type(sm) is np.str_:
        mol = Chem.MolFromSmiles(sm)
        if '.' in sm: use_coordgen = True # only this mode handles disconnected parts correctly
    else:
        mol = sm

    # 50% chance to draw explicit hydrogens where they make sence (mostly in stereo) 
    if toss_coin():
        mol = Chem.AddHs(mol, explicitOnly=True)

    # CoordGen call is disabled, see above
    #rdDepictor.SetPreferCoordGen(use_coordgen)
    
    if shape is None: # guess canvas shape
        molsize = mol.GetNumAtoms()
        shape=(150+molsize*15, 150+molsize*15)

    if len(varbonds) > 0:
        svg = True
    
    if not svg:
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(*shape)
    else:
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(*shape)

    # coordgen mode uses own scaling.
    # experimentally it was found that a magical coefficient 0.7 fixes the difference
    if use_coordgen:
        line_length /= 0.7
        font_size *= 0.7
        atom_padding *= 0.7

    # fixed line width is given as function parameter.
    # New rdkit drawer requires absolute values of font size (previously - percental values, which was very convinient)
    # This function accepts percental font size and then multiplied on given bond length, which gives very stables scaling
    font_size = int(font_size*line_length)
    
    drawer_opts = drawer.drawOptions()
    drawer_opts.minFontSize = font_size
    drawer_opts.maxFontSize = font_size
    if font_file is not None:
        drawer_opts.fontFile = font_file
    
    drawer_opts.fixedBondLength = int(line_length)

    if toss_coin(0.05):
        drawer_opts.addStereoAnnotation = True

    # Almost all images in papers are black&white, and we draw it in 95% cases
    # Sometimes they draw colored molecules in papers (here in 5%).
    # Later we will shuffle colormap and convert to grayscale
    is_blackwhite = toss_coin(0.95)
    if is_blackwhite:
        drawer_opts.useBWAtomPalette()
    else:
        drawer_opts.useDefaultAtomPalette()
    
    drawer_opts.padding = 0
    drawer_opts.multipleBondOffset = float(multiple_bond_offset)
    drawer_opts.rotate = int(rotate)
    drawer_opts.bondLineWidth = int(line_width)
    drawer_opts.additionalAtomLabelPadding = float(atom_padding)

    # 50/50 generate [2H],[3H] and D,T images
    if toss_coin():
        drawer_opts.atomLabelDeuteriumTritium = True
    
    mol2draw = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
    drawer.DrawMolecule(mol2draw)
    drawer.FinishDrawing()

    # variate position augment returns ids of bond pairs to unite (see vargroup_aug.py)
    # if some varbonds are given, draw as svg and fix svg manually - unite these bond pairs
    # the procedure exploits undocumented features and is unstable.
    # In some cases (<5%) it fails, so an error is raised, it is catched later
    if not svg:
        png = bytearray(drawer.GetDrawingText())
    else:
        svg = drawer.GetDrawingText()
        if len(varbonds) > 0:
            try:
                svg = svg_fix_vargroup(svg, varbonds)
            except:
                raise Exception("Cannot fix vargroup")
        png = surface.PNGSurface.convert(svg)
    
    pic = PIL.Image.open(io.BytesIO(png))
    img = np.array(pic)
    drawer.ClearDrawing()
    
    return img


# the rdkit drawer function contains few minor variability (black&white, stereo annotation, D&T)
# main variablity is given in this function
# `no_coordgen` option restricts coordgen usage, it was added to walkaround the EZ segfault case (see above)
def draw_mol_rdkit_aug (sm, svg=False, varbonds=[], no_coordgen=False):
    # rotate to fixed (0,90,180,270) or random angle
    rotate_opt = np.random.choice([0,0,0,1,2,2,3,4,4])
    if rotate_opt == 4:
        rotate = np.random.randint(360)
    else:
        rotate = int(rotate_opt*90)

    # 10% chance to set random external fond
    if toss_coin(0.1):
        font_file = font_paths[np.random.choice(len(font_paths))]
    else:
        font_file = None

    if no_coordgen:
        use_coordgen = False
    else:
        use_coordgen = toss_coin(0.33)

    # value ranges of other params given here
    kwargs = {"font_size" : np.random.uniform(0.65, 1.1),
              "font_file" : font_file,
              "line_width" : np.random.choice([1]*3+[2]*6+[3]*1),
              "line_length" : np.random.randint(18,25),
              "multiple_bond_offset" : np.random.uniform(0.08, 0.18),
              "rotate" : rotate,
              "atom_padding" : np.random.uniform(0, 0.3),
              "use_coordgen" : use_coordgen }
    
    return draw_mol_rdkit(sm, shape=None, svg=svg,
                    varbonds=varbonds, **kwargs)

################################################################

'''
We use fixed bond width mode, so if given shape is bigger than required, whitespace is around the molecule. 
This function standardizes the whitespace size (or padding)
First it removes all padding - crop bounding box of nonzero elements.
Then is adds whitespace according to given padding value
The formula `padding*sqrt(molsize)*10 + 50` is empirical.
If real molsize is too small (like 1 or 2 atoms, or an empty ring), the percental padding becomes invisible,
unless we add a fixed pad like 50 pixels
sqrt(size)*10 is used instead of simple size*padding, because for very big molecules (like 1000 pixels and more)
padsize is also very big.

All images are padded to sqare shape by max(width, height)
'''

def fix_padding (img, padding):
    # ensure background color is pure white
    bg_color = np.ceil(np.median(img, axis=[0,1])).astype(np.uint8)
    bg_diff = 255-bg_color
    img = img+bg_diff
    
    pic = PIL.Image.fromarray(img)
    bbox = PIL.ImageChops.invert(pic).getbbox()
    if bbox is None:
        raise Exception("Empty image")
    
    bbox = list(bbox)
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    
    maxdim = int(max(h,w))
    h_pad = (maxdim-h)//2
    w_pad = (maxdim-w)//2
    bbox[0] -= w_pad
    bbox[2] += w_pad
    bbox[1] -= h_pad
    bbox[3] += h_pad
    
    pic = pic.crop(tuple(bbox))
    
    padsize = int((maxdim**0.5)*10*padding)+50
    pic = ImageOps.expand(pic, border=padsize, fill=(255,255,255))
    
    return np.array(pic)

# Other drawing options are implemented, but they require to know drawing shape.
# This function exploits rdkit drawer to guess drawing shape for a molecule
def rdkit_guess_shape (sm):
    mol = Chem.MolFromSmiles(sm)
    molsize = len(mol.GetAtoms())
    fallback_shape = 150+molsize*10
    min_shape = fallback_shape//2
    try:
        rdkit_img = draw_mol_rdkit(sm)
        rdkit_img = fix_padding(rdkit_img, padding=0)
        h,w = rdkit_img.shape[:2]
        shape = (h+w)//2
    except Exception as e:
        # if failed to draw a mol (valence problems or smth), no reason to deny the molecule,
        # just return fallback shape guess and try if another drawer can work 
        print(f'Fallback shape est for {sm}: {str(e)}')
        shape = fallback_shape

    # some shape variety
    random_scale = np.random.uniform(0.75, 1.15)
    shape = int(shape*random_scale)
    shape = max(shape, min_shape)
    return shape

# check if the image is black&white
def is_blackwhite (img):
    diff1 = np.mean(np.abs(img[:,:,0] - img[:,:,1]))
    diff2 = np.mean(np.abs(img[:,:,1] - img[:,:,2]))
    diff3 = np.mean(np.abs(img[:,:,0] - img[:,:,2]))
    diff = (diff1+diff2+diff3)/3
    return diff < 0.01

# colored to grayscale
# if not black&white - random colormap shuffle
def decolorize_pil (img):
    if not is_blackwhite(img):
        if toss_coin():
            swapcolors = np.random.permutation([0,1,2])
            img = img[:,:,swapcolors]
    
    pic = PIL.Image.fromarray(img)
    return np.array(pic.convert('L'))

################################################################

'''
Pubchem drawer. Its drawing style differs from rdkit one, so could be used in small amount.
Disabled by _USE_PUBCHEM_ var in head, not to spam while testing.
Enable if want to use it
If shape not givem guess with rdkit drawer (see above)
guess shape is modified as *0.7 +150, empirically fixing own pubchem padding
'''
def draw_mol_pubchem (sm, shape=None):
    if shape is None:
        shape = rdkit_guess_shape(sm)
        shape = int(shape*0.7 + 150)
    
    fname = np.random.randint(1e16)
    fname = f"tmpdir/{str(hex(fname))[2:]}.png"
    if not os.path.exists("tmpdir"):
        os.mkdir("tmpdir")
    command = f'curl -d "smiles={sm}" -X POST https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/png?image_size={shape}x{shape} --output {fname} 2>/dev/null'
    system_call(command, 20)
    img = np.array(PIL.Image.open(fname).convert('RGB'))
    os.remove(fname)
    if img is None:
        raise Exception("Pubchem load failed")

    # pubchem returns colored images
    # in 90% - to black&white 3-channel
    # thresholding is used to make pure b&w
    # thr randomness gives variety of width and drawing intensity 
    if toss_coin(0.9):
        img_gs = np.min(img, axis=2)
        thr = np.random.randint(130, 180+1)
        img_gs = ((img_gs>thr)*255).astype(np.uint8)
        for i in [0,1,2]:
            img[:,:,i] = img_gs
    return img

'''
OpenBabel drawer. Not very brilliant, but can bring drawing variety
Totally tolerant to valence problems. Can easily draw 10-valent F.
Drawing shape guessed by rdkit and fixed empirically, because it does own padding
Drawint time grows exponentially after 50 atoms (or simply 70 smiles chars)
'''
def draw_mol_openbabel (sm, shape=None):
    if len(sm) > 70: raise Exception("Large molecule, use rdkit instead")
    if shape is None:
        shape = rdkit_guess_shape(sm)
        shape = int(shape*0.7 + 50)
    
    fname = np.random.randint(1e16)
    fname = f"tmpdir/{str(hex(fname))[2:]}.png"
    flags = f"-xu -xs -xp{shape}"

    if not os.path.exists("tmpdir"):
        os.mkdir("tmpdir")
    
    # draw explicit carbons in 0.01% cases
    if toss_coin(0.0001):
        flags += ' -xa'
    else:
        # draw explicit hydrogens in end carbons in 1% cases
        if toss_coin(0.99):
            flags += ' -xC'
    
    command = f"obabel -:'{sm}' -O {fname} {flags} 2> /dev/null"
    # drawing time exponentially depends on number of atoms,
    # after 60 atoms it can take minutes, so giving 20 seconds timeout
    system_call(command, 20)
    img = np.array(PIL.Image.open(fname).convert('RGB'))
    os.remove(fname)
    
    return img

################################################################

'''
Toplevel drawing function
Defines if to use rdkit (default) or try other drawers.
rdkit is used if
1) rdkit mol is given (not a smiles string), meaning it could be FGMol
2) svg required
3) varbonds are given
 
In our generator FGMol is generated in about 90% cases.
Alternative drawer is called used in 50% of other cases, meaning 5% of total cases.
'''

def draw_mol_aug (src, padding=0.2, svg=False, varbonds=[], no_coordgen=False):
    use_rdkit = True
    # if got regular smiles, not svg and no var bonds,
    # give a probability for alternative drawers
    if type(src) is str and \
       not svg and len(varbonds) == 0 and \
       Chem.MolFromSmiles(src) is not None:
        if toss_coin(0.5):
            use_rdkit = False
            
    if not use_rdkit:
        try:
            # pubchem to openbabel usage is 1/3.
            # If you are sure pubchem or provider not gonna ban you, try set 50/50
            if _USE_PUBCHEM_ and toss_coin(0.25):
                img = draw_mol_pubchem(src)
            else:
                img = draw_mol_openbabel(src)
        except:
            use_rdkit = True

    if use_rdkit:
        img = draw_mol_rdkit_aug(src, svg=svg, varbonds=varbonds, no_coordgen=no_coordgen)
        
    img = fix_padding(img, padding=padding)
    return img
            
        
