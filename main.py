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
#    hp_seq = hp2d.string_seq_to_bin("HHPPHPPHPPHPPHPPHPPHPPHH")
#    len_hp_seq = hp_seq.shape[0]
#    print(len_hp_seq)
##    x = np.random.uniform(lower_bound,upper_bound,len_hp_seq-2) #x should be generated from optimization algorithms
##    print(hp2d.func_2D_protein(x, hp_seq, settings))
#    bounds = [(lower_bound, upper_bound)]*(len_hp_seq-2)
#    #opt algo
#    result = differential_evolution(hp2d.func_2D_protein, bounds, (hp_seq, settings), strategy='best1bin', disp=True, popsize=len_hp_seq*20, polish=True, workers=-1, maxiter=750)
#    print(result)
#    print("coor = ", hp2d.to_2D_cartesian_from_real(result.x, settings))
#    
#    ret_val = np.array(hp2d.return_val)
#    np.save("data/UM24_data", hp2d.return_val, allow_pickle=True)
#    ret_v = np.load("data/UM24_data.npy", allow_pickle=True)
#    print(ret_v, ret_v.shape)
#    end = time.time()
#    print("elapsed time = ", end-start)
#    
    
    sequences = ["PPHPHHHPHHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
                 "HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
                 "PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHH",
                 ]
    files = ["UM48", "UM50", "UM60"]
    for i in range(len(sequences)):
        start = time.time()    #timer
        hp_seq = hp2d.string_seq_to_bin(sequences[i])
        len_hp_seq = hp_seq.shape[0]
        print(len_hp_seq)
        bounds = [(lower_bound, upper_bound)]*(len_hp_seq-2)
        result = differential_evolution(hp2d.func_2D_protein, bounds, (hp_seq, settings), strategy='best1bin', disp=True, popsize=len_hp_seq*20, polish=True, workers=-1, maxiter=1)
        print(result)
        print("coor = ", hp2d.to_2D_cartesian_from_real(result.x, settings))
        ret_val = np.array(hp2d.return_val)
        np.save("data/"+files[i]+"_data", hp2d.return_val, allow_pickle=True)
        ret_v = np.load("data/"+files[i]+"_data.npy", allow_pickle=True)
        print(ret_v, ret_v.shape)
        end = time.time()
        print("elapsed time = ", end-start)