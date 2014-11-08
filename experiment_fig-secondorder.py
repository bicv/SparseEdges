import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges

pe = ParameterSet('default_param.py')
im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)

for name_database in ['serre07_distractors', 'laboratory']:
    # control experiment
    imageslist, edgeslist, RMSE = mp.process(exp='prior_vanilla', name_database=name_database)
    imageslist, edgeslist_noise, RMSE = mp.process(exp='prior_vanilla_noise', name_database=name_database, noise=pe.noise)

    try:
        mp.pe.eta_SO = .5
        imageslist, edgeslist, RMSE =  mp.process(exp='prior_vanilla_secondorder', name_database=name_database)
        imageslist, edgeslist, RMSE = mp.process(exp='prior_vanilla_secondorder_noise', name_database=name_database, noise=pe.noise)
    except:
        print('run again once first batches are finished ')