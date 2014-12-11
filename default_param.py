# -*- coding: utf8 -*-
{
# Image
# 'N_image' : None, #use all images in the folder
'N_image' : 100, #use 100 images in the folder
# 'N_image' : 10, #use 4 images in the folder
'seed' : None, # a seed for the Random Number Generator (RNG) for picking images in databases, set to None xor set to a given number to freeze the RNG
'N_X' : 256, # size of images
'N_Y' : 256, # size of images
# 'N_X' : 64, # size of images
# 'N_Y' : 64, # size of images
'noise' : 0.2, # level of noise when we use some
'do_mask' : True, # used in SLIP
# whitening parameters:
'do_whitening'  : True, # = self.pe.do_whitening
'white_name_database' : 'serre07_distractors',
'white_n_learning' : 0,
'white_N' : .07,
'white_N_0' : .0, # olshausen = 0.
'white_f_0' : .4, # olshausen = 0.2
'white_alpha' : 1.4,
'white_steepness' : 4.,
'white_recompute' : False,
# Log-Gabor
#'base_levels' : 2.,
'base_levels' : 1.618,
'n_theta' : 24, # number of (unoriented) angles between 0. radians (included) and np.pi radians (excluded)
'B_sf' : .4, # 1.5 in Geisler
'B_theta' : 3.14159/18.,
# Matching Pursuit
# 'N' : 32, # number of edges extracted
'N' : 2**11,
# 'N' : 2**8,
'MP_alpha' : .7, # ratio of inhibition in alpha-Matching Pursuit
'MP_rho' : None, # geometric scaling parameter
'eta_SO' : 0., # including a dipole
'MP_do_mask'  : True, # used in Matching Pursuit self.pe.do_mask
# parameters for computing the histograms
'd_width' : 45., # Geisler 1.23 deg (full image = 45deg)
'd_min' : .25, # Geisler 1.23 deg (full image = 45deg)
'd_max' : 2., # Geisler 1.23 deg (full image = 45deg)
'N_r' : 6, #
'N_Dtheta' : 24, # equal to n_theta : 24 to avoid artifacts
'N_phi' : 12, #
'N_scale' : 5, #
'loglevel_max': 7, # used for the statistics
'figsize_hist' : 3.41, # width of a column in inches
'figsize_cohist' : 3.41, #
# doing the computation on a circular mask
'edge_mask' : True, #
'do_rank': False,
'scale_invariant': True,
'multiscale': True,
'kappa_phase': 0.,
'weight_by_distance': True,
# Dipole
'dip_w':.1,
'dip_B_psi':.4,
'dip_B_theta':.8,
'dip_scale':1.5,
'dip_epsilon':5.e-1,
# PATHS
'figpath' : 'figures/',
'edgefigpath' : 'figures/edges/',
'matpath' : 'mat/',
'edgematpath' : 'mat/edges/',
'datapath' : '../AssoField/database/',
'ext' : '.pdf',
'scale' : .2,
'scale_circle' : 0.08, # relativesize of segments and pivot
'scale_chevrons' : 2.5,
'line_width': 1.,
'line_width_chevrons': .75,
'edge_scale_chevrons': 180.,
'figsize_edges' : 6,
}
