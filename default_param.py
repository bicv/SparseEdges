{
# COMPUTATIONAL PARAMETERS
# 'ncpus' : 'autodetect', # for a SMP machine
#'ncpus' : 16, # Cluster
'ncpus' : 1, # on the cluster we can run many batches - no need for pp
'seed' : None, # seed used in the number generator for collecting images etc...
# Image
'N_X' : 256, # size of images
'N_Y' : 256, # size of images
# 'N_X' : 64, # size of images
# 'N_Y' : 64, # size of images
'do_mask'  : True, # used in MAtching Pursuit self.pe.do_mask
'seed': None, # a seed for the Random Number Generator (RNG) for picking images in databases, set to None or a given number to freeze the RNG
# Log-Gabor
#'base_levels' : 2.,
'base_levels' : 1.618,
'n_theta' : 24, # number of (unoriented) angles between 0. radians (included) and np.pi radians (excluded)
'B_sf' : .5, # 1.5 in Geisler
'B_theta' : 3.14159/12.,
# Matching Pursuit
# TODO : use 1 ??
'alpha' : .0, # exponent of the color envelope
'MP_alpha' : .8, # ratio of inhibition in alpha-Matching Pursuit
# 'N' : 32, # number of edges extracted
'N' : 2**11,
'do_whitening'  : True, # = self.pe.do_whitening
'MP_do_mask'  : False, # used in MAtching Pursuit self.pe.do_mask
#do_real=False # do we consider log-gabors with a complex part?
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
# parameters for computing the histograms
'd_width' : 45., # Geisler 1.23 deg (full image = 45deg)
'd_min' : .25, # Geisler 1.23 deg (full image = 45deg)
'd_max' : 2., # Geisler 1.23 deg (full image = 45deg)
'N_r' : 6, #
'N_Dtheta' : 24, # equal to n_theta : 24 to avoid artifacts
'N_phi' : 12, #
'N_scale' : 5, #
'loglevel_max': 7,
'alpha' : .0, # exponent of the color envelope
'N_image' : 100, #use all images in the folder 200, #None
# 'N_image' : 4, #use all images in the folder 200, #None
'noise' : 0.5, #
'figsize_hist' : 3.41, # width of a column in inches
'figsize_cohist' : 3.41, #
# doing the computation on a circular mask
'edge_mask' : True, #
'do_rank': False,
'scale_invariant': True,
'multiscale': True,
'kappa_phase': 0.,
'weight_by_distance': True,
}
