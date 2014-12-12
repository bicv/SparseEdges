
# rm **/Geisler01Fig7A_secondorder*
# rm **/circle_in_noise_secondorder*

import os
import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges

#figpath = '../../CNRS/BICV-book/BICV_INT/BICV-sparse/'
figpath = './'
FORMATS = ['pdf', 'eps']
fig_width_pt = 318.670  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches


def init_pe():
    pe = ParameterSet('default_param.py')
    pe.N = 36
    pe.do_whitening = True
    pe.MP_alpha = 1.
    pe.figsize_edges = 12
    pe.figsize_edges = .382 * fig_width
    pe.scale = 1.3
    pe.line_width = 1.5
    return pe

pe = init_pe()
eta_SO = 0.3
im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)

##############################################################################################################

if False:

    figname = 'circle_in_noise' # Geisler01Fig7A_rec
    # defining input image 
    from pylab import imread
    image = imread('database/' + figname + '.png').mean(axis=-1)
    print image.mean(), image.std()
else:
    N, N_circle, N_image = 1024, 36, 1
    edgeslist = np.zeros((6, N+N_circle, N_image))
    # random edges:
    edgeslist[0, :N, :] = pe.N_X * np.random.rand(N, N_image)
    edgeslist[1, :N, :] = pe.N_X * np.random.rand(N, N_image)
    edgeslist[2, :N, :] = (np.pi* np.random.rand(N, N_image) ) % np.pi
    edgeslist[3, :N, :] = 0.5 * (1- pe.base_levels**(-mp.n_levels*(np.random.rand(N, N_image))))
    edgeslist[4, :N, :] = 1.2*np.random.rand(N, N_image) * np.sign(np.random.randn(N, N_image))
    edgeslist[5, :N, :] = 2*np.pi*np.random.rand(N, N_image)
    # cocircular edges:
    for i_N, angle in enumerate(np.linspace(0, 2*np.pi, N_circle)): #2*np.pi*np.random.rand(N_circle)):
        edgeslist[0, N + i_N, :] = pe.N_X/2. - pe.N_X/4.*np.sin(angle) + .0 * np.random.randn(N_image)
        edgeslist[1, N + i_N, :] = pe.N_X/2. + pe.N_X/4.*np.cos(angle) + .0 * np.random.randn(N_image)
        edgeslist[2, N + i_N, :] = (np.pi/2 + angle + .5*np.pi/180 * np.random.randn(N_image)) % np.pi
        edgeslist[3, N + i_N, :] = mp.sf_0[2] #0.03
        edgeslist[4, N + i_N, :] = .8 + .3*np.exp(np.cos(angle)/1.**2)

    print edgeslist.shape
    image = mp.reconstruct(edgeslist[:,:,0])
    from pylab import imsave, gray
    imsave(fname='database/circle_in_noise.png', arr=image, vmin=image_rec.min(), vmax=image_rec.max(), cmap=gray())

image = im.normalize(image, center=True)
print image.mean(), image.std()
v_max = 1.*image.max()
v_min = -v_max
    
##############################################################################################################
print ' without second-order '
matname = 'mat/' + figname + '_secondorder_A.npy'
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
        for ext in FORMATS: 
                fig.savefig(matname.replace('mat/', mp.pe.figpath).replace('.npy', '.' + ext))
except:
    print 'File ', matname, ' is locked'

##############################################################################################################
print ' with second-order '
matname = 'mat/' + figname + '_secondorder_B.npy'
if not(os.path.isfile(matname)):
    if not(os.path.isfile(matname + '_lock')):
        file(matname + '_lock', 'w').close()
        mp.pe.eta_SO = eta_SO
        edges, C_res = mp.run_mp(image, verbose=True)
        np.save(matname, edges)
        os.remove(matname + '_lock')
try:
    edges = np.load(matname)
    edges[4, :] *= -1 # turn red in blue...
    fig, a = mp.show_edges(edges, image=image, v_min=v_min, v_max=v_max, color='toto', show_phase=False) #
    if not(figpath==None): 
        for ext in FORMATS: 
                fig.savefig(matname.replace('mat/', mp.pe.figpath).replace('.npy', '.' + ext))
except:
    print 'File ', matname, ' is locked'

    
##############################################################################################################
N_explore = 25
base = 4.
##############################################################################################################
mp = SparseEdges(LogGabor(Image(init_pe())))
for mp.pe.eta_SO in np.logspace(-1., 1., N_explore, base=base)*eta_SO:
    matname = 'mat/' + figname + '_secondorder_eta_SO_' + str(mp.pe.eta_SO).replace('.', '_') + '.npy'
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
        print 'File ', matname, ' is locked'

##############################################################################################################
mp = SparseEdges(LogGabor(Image(init_pe())))
mp.pe.eta_SO = eta_SO
for mp.pe.dip_w in np.logspace(-1., 1., N_explore, base=base)*pe.dip_w:
    matname = 'mat/' + figname + '_secondorder_dip_w_' + str(mp.pe.dip_w).replace('.', '_') + '.npy'
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
        print 'File ', matname, ' is locked'

##############################################################################################################
mp = SparseEdges(LogGabor(Image(init_pe())))
mp.pe.eta_SO = eta_SO
for mp.pe.dip_epsilon in np.logspace(-1., 1., N_explore, base=base)*pe.dip_epsilon:
    matname = 'mat/' + figname + '_secondorder_dip_epsilon_' + str(mp.pe.dip_epsilon).replace('.', '_') + '.npy'
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
        print 'File ', matname, ' is locked'
        
##############################################################################################################
mp = SparseEdges(LogGabor(Image(init_pe())))
mp.pe.eta_SO = eta_SO
for mp.pe.dip_B_psi in np.logspace(-1., 1., N_explore, base=base)*pe.dip_B_psi:
    matname = 'mat/' + figname + '_secondorder_dip_B_psi_' + str(mp.pe.dip_B_psi).replace('.', '_') + '.npy'
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
        print 'File ', matname, ' is locked'

##############################################################################################################
mp = SparseEdges(LogGabor(Image(init_pe())))
mp.pe.eta_SO = eta_SO
for mp.pe.dip_B_theta in np.logspace(-1., 1., N_explore, base=base)*pe.dip_B_theta:
    matname = 'mat/' + figname + '_secondorder_dip_B_theta_' + str(mp.pe.dip_B_theta).replace('.', '_') + '.npy'
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
        print 'File ', matname, ' is locked'
##############################################################################################################