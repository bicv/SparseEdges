
# rm **/Geisler01Fig7A*

import os
import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges
pe = ParameterSet('default_param.py')
pe.N = 210
pe.N = 105
pe.do_whitening = False
#pe.do_whitening = True


#figpath = '../../CNRS/BICV-book/BICV_INT/BICV-sparse/'
figpath = './'
FORMATS = ['pdf', 'eps']
fig_width_pt = 318.670  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches
pe.figsize_edges = 12
pe.figsize_edges = .382 * fig_width
pe.scale = 1.3
pe.line_width = 1.5


# defining input image 
from pylab import imread
image = imread('database/Geisler01Fig7A.png').mean(axis=-1)
print image.mean(), image.std()

im = Image(pe)
image = im.normalize(image, center=True)
print image.mean(), image.std()
v_max = 1.*image.max()
v_min = -v_max

lg = LogGabor(im)
mp = SparseEdges(lg)
print mp.n_levels, mp.sf_0

print ' without second-order '
matname = 'mat/Geisler01Fig7A.npy'
if not(os.path.isfile(matname)):
    if not(os.path.isfile(matname + '_lock')):
        file(matname + '_lock', 'w').close()
        mp.pe.eta_SO = 0.
        edges, C_res = mp.run_mp(image, verbose=True)
        np.save(matname, edges)
        os.remove(matname + '_lock')
try:
    edges = np.load(matname)
    fig, a = mp.show_edges(edges, image=image, v_min=v_min, v_max=v_max, color='toto', show_phase=False) #
    if not(figpath==None): 
        for ext in FORMATS: fig.savefig(figpath + 'Geisler01Fig7_secondorder_A.' + ext)
except:
    print 'Failed with ', matname

    

print ' with second-order '
matname = 'mat/Geisler01Fig7A_secondorder.npy'
if not(os.path.isfile(matname)):
    if not(os.path.isfile(matname + '_lock')):
        file(matname + '_lock', 'w').close()
        mp.pe.eta_SO = 0.75
        edges, C_res = mp.run_mp(image, verbose=True)
        np.save(matname, edges)
        os.remove(matname + '_lock')
try:
    edges = np.load(matname)
    edges[4, :] *= -1 # turn red in blue...
    fig, a = mp.show_edges(edges, image=image, v_min=v_min, v_max=v_max, color='toto', show_phase=False) #
    if not(figpath==None): 
        for ext in FORMATS: fig.savefig(figpath + 'Geisler01Fig7_secondorder_B.' + ext)
except:
    print 'Failed with ', matname

im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)
for mp.pe.eta_SO in np.linspace(.0, .5, 9):
    matname = 'mat/Geisler01Fig7A_secondorder_eta_SO_' + str(mp.pe.eta_SO).replace('.', '_') + '.npy'
    if not(os.path.isfile(matname)):
        if not(os.path.isfile(matname + '_lock')):
            file(matname + '_lock', 'w').close()
            edges, C_res = mp.run_mp(image, verbose=True)
            np.save(matname, edges)
            os.remove(matname + '_lock')
    try:
        edges = np.load(matname)
        edges[4, :] *= -1 # turn red in blue...
        fig, a = mp.show_edges(edges, image=image, v_min=v_min, v_max=v_max, color='toto', show_phase=False) #
        fig.savefig(matname.replace('mat/', mp.pe.figpath).replace('.npy', '.pdf'))
    except:
        print 'Failed with ', matname

        
im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)
for mp.pe.dip_w in np.linspace(.01, .5, 9):
    matname = 'mat/Geisler01Fig7A_secondorder_dip_w_' + str(mp.pe.dip_w).replace('.', '_') + '.npy'
    if not(os.path.isfile(matname)):
        if not(os.path.isfile(matname + '_lock')):
            file(matname + '_lock', 'w').close()
            edges, C_res = mp.run_mp(image, verbose=True)
            np.save(matname, edges)
            os.remove(matname + '_lock')
    try:
        edges = np.load(matname)
        edges[4, :] *= -1 # turn red in blue...
        fig, a = mp.show_edges(edges, image=image, v_min=v_min, v_max=v_max, color='toto', show_phase=False) #
        fig.savefig(matname.replace('mat/', mp.pe.figpath).replace('.npy', '.pdf'))
    except:
        print 'Failed with ', matname

im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)
for mp.pe.dip_B_psi in np.linspace(.01, 2.5, 25):
    matname = 'mat/Geisler01Fig7A_secondorder_dip_B_psi_' + str(mp.pe.dip_B_psi).replace('.', '_') + '.npy'
    if not(os.path.isfile(matname)):
        if not(os.path.isfile(matname + '_lock')):
            file(matname + '_lock', 'w').close()
            edges, C_res = mp.run_mp(image, verbose=True)
            np.save(matname, edges)
            os.remove(matname + '_lock')
    try:
        edges = np.load(matname)
        edges[4, :] *= -1 # turn red in blue...
        fig, a = mp.show_edges(edges, image=image, v_min=v_min, v_max=v_max, color='toto', show_phase=False) #
        fig.savefig(matname.replace('mat/', mp.pe.figpath).replace('.npy', '.pdf'))
    except:
        print 'Failed with ', matname

im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)
for mp.pe.dip_B_theta in np.linspace(.01, 1.5, 25):
    matname = 'mat/Geisler01Fig7A_secondorder_dip_B_theta_' + str(mp.pe.dip_B_theta).replace('.', '_') + '.npy'
    if not(os.path.isfile(matname)):
        if not(os.path.isfile(matname + '_lock')):
            file(matname + '_lock', 'w').close()
            edges, C_res = mp.run_mp(image, verbose=True)
            np.save(matname, edges)
            os.remove(matname + '_lock')
    try:
        edges = np.load(matname)
        edges[4, :] *= -1 # turn red in blue...
        fig, a = mp.show_edges(edges, image=image, v_min=v_min, v_max=v_max, color='toto', show_phase=False) #
        fig.savefig(matname.replace('mat/', mp.pe.figpath).replace('.npy', '.pdf'))
    except:
        print 'Failed with ', matname