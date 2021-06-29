
from copy import deepcopy
import xml
import numpy as np

'''
Some write-only code to edit svg with variated position bonds.
Main idea is to find paths with class 'bond-X' in given svg, remove them
and draw a new path at the middle of the deleted pair. See visual examples to get the idea.
This code tries to simulate different variants of drawing - lengths, gaps, shifts, etc.
'''

def get_bonds (tree):
    bonds = []
    for e in tree.iter():
        if 'path' in e.tag:
            dct = dict(e.items())
            if 'bond' in dct['class']:
                bonds.append({'src' : e,
                              'dict' : dct,
                              'parsed' : parse_bond(dct)})
    return bonds

def parse_bond (b):
    N = int(float(b['class'].split('-')[1]))
    M,L = b['d'].split('M')[1].split('L')
    M,L = [np.array([float(x) for x in p.split(',')]) for p in [M,L]]
    return N,M,L

def make_d (p1, p2):
    return f' M {p1[0]:.3f},{p1[1]:.3f} L {p2[0]:.3f},{p2[1]:.3f}'

def white_style (style, gap):
    params = style.split(';')
    for i,p in enumerate(params):
        k,v = p.split(':')
        if k == 'stroke-width':
            v = int(float(v.split('px')[0]))*gap #gap = np.random.randint(8,16)
            params[i] = f'stroke-width:{v}px'
        if k == 'stroke':
            params[i] = 'stroke:#FFFFFF'
    return ';'.join(params)

def distance (p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def uf (a, b):
    return np.random.uniform(a, b)

def svg_fix_vargroup (svg, r_bonds):
    tree = xml.etree.ElementTree.fromstring(svg)
    bonds = get_bonds(tree)
    for b1id,b2id in r_bonds:
        b1s = [b for b in bonds if b['parsed'][0] == b1id]
        b2s = [b for b in bonds if b['parsed'][0] == b2id]
        b1 = b1s[0]
        b2 = b2s[0]
        n1,p11,p12 = b1['parsed']
        n2,p21,p22 = b2['parsed']
        combinations = [[p11,p21,p12,p22], [p11,p22,p12,p21], [p12,p21,p11,p22], [p12,p22,p11,p21]]
        p1r,p2r,p1c,p2c = combinations[np.argmin([distance(*c[:2]) for c in combinations])]
        pr = (p1r+p2r)/2
        pc = (p1c+p2c)/2
        pc_fix = pc-uf(0.4, 0.6)*(pr-pc)
        pr_fix = pr-uf(0.2, 0.5)*(pr-pc)
        d = make_d(pr_fix,pc_fix)

        if len(b1s) + len(b2s) == 3: # crossing double line
            # black r line
            brb = deepcopy(b1['src'])
            brb.set('class', f'bond-r{n1}{n2}')
            brb.set('d', d)
            tree.append(brb)

            # white gap
            bcw = deepcopy(b1['src'])
            bcw.set('class', f'bond-c{n1}{n2}')
            bcw_p1, bcw_p2 = p1c-(p1c-p2c)*0.4, p1c-(p1c-p2c)*0.6
            bcw_d = make_d(bcw_p1, bcw_p2)
            bcw.set('d', bcw_d)
            whitegap = distance(p1c,p2c)*uf(0.1, 0.15)
            bcw.set('style', white_style(bcw.get('style'), whitegap))
            tree.append(bcw)
            
           
            # add double line to ring bond + white gap
            rc = pc-(pr-pc)
            shift = uf(0.2, 0.28)
            p1c2 = p1c-shift*(p1c-rc)
            p2c2 = p2c-shift*(p2c-rc)
            
            bcw2 = deepcopy(b1['src'])
            bcw2.set('class', f'bond-c{n1}{n2}')
            p1c2w, p2c2w = p1c2-(p1c2-p2c2)*0.37, p1c2-(p1c2-p2c2)*0.63
            bcw2_d = make_d(p1c2w, p2c2w)
            bcw2.set('d', bcw2_d)
            whitegap = distance(p1c,p2c)*uf(0.08, 0.12)
            bcw2.set('style', white_style(bcw2.get('style'), whitegap))
            tree.append(bcw2)
            
            bcb2 = deepcopy(b1['src'])
            bcb2.set('class', f'bond-c{n1}{n2}')
            bcb2_d = make_d(p1c2, p2c2)
            bcb2.set('d', bcb2_d)
            tree.append(bcb2)
           
            # restore ring bond
            bcb = deepcopy(b1['src'])
            bcb.set('class', f'bond-c{n1}{n2}')
            bcb_d = make_d(p1c, p2c)
            bcb.set('d', bcb_d)
            tree.append(bcb)
            
        
        else:
            brw = deepcopy(b1['src'])
            brw.set('class', f'bond-r{n1}{n2}')
            brw.set('d', d)
            brb = deepcopy(brw)
            whitegap = int(distance(p1c, p2c)*uf(0.15, 0.3))
            brw.set('style', white_style(brw.get('style'), whitegap) )
            tree.append(brw)
            tree.append(brb)
        
        for b1 in b1s:
            tree.remove(b1['src'])
        for b2 in b2s:
            tree.remove(b2['src'])
        
    svg = xml.etree.ElementTree.tostring(tree).decode('utf-8')
    return svg
