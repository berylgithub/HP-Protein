# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:16:05 2020

@author: Saint8312
"""

import time
import numpy as np
from scipy.optimize import differential_evolution

import HP_Protein_2D as hp2d

if __name__ == "__main__":
    start = time.time()    #timer
    
    np.random.seed(13) #set seed if necessary
    #inputs
    settings={
            "E_contact": {"HH":-1., "HP":0., "PH":0., "PP":0.},
            "ranges":np.array(((0,3),(3,6),(6,9)))
            }
    lower_bound = settings["ranges"][0][0]
    upper_bound = settings["ranges"][settings["ranges"].shape[0]-1][settings["ranges"].shape[1]-1]
    hp_seq = hp2d.string_seq_to_bin("HHPPHPPHPPHPPHPPHPPHPPHH")
    len_hp_seq = hp_seq.shape[0]
    x = np.random.uniform(lower_bound,upper_bound,len_hp_seq-2) #x should be generated from optimization algorithms
    print(hp2d.func_2D_protein(x, hp_seq, settings))
    bounds = [(lower_bound, upper_bound)]*(len_hp_seq-2)
    #opt algo
    result = differential_evolution(hp2d.func_2D_protein, bounds, (hp_seq, settings), maxiter=250, strategy='best1bin', disp=True, popsize=len_hp_seq*20, workers=1)
    print(result)
    print("coor = ", hp2d.to_2D_cartesian_from_real(result.x, settings))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    end = time.time()
    print("elapsed time = ", end-start)