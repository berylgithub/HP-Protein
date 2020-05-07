# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:16:05 2020

@author: Saint8312
"""

import time
import numpy as np
from scipy.optimize import differential_evolution
from differential_evolution import diff_evol, diff_evol_max
from Clustering import cluster_DE, mat_R_ij, transformation_matrix
import multiprocessing
import pickle


import HP_Protein_2D as hp2d

if __name__ == "__main__":
    start = time.time()    #timer
    
    np.random.seed(13) #set seed if necessary
    pool = multiprocessing.Pool()
    
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
    
    sequences = ["HPHPPHHPHPPHPHHPPHPH",
                 "HHPPHPPHPPHPPHPPHPPHPPHH",
                 "PPHPPHHPPPPHHPPPPHHPPPPHH",
                 "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",
                 "PPHPHHHPHHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
                 "HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
                 "PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHH",
                 "HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHHH",
                 "HPPHPH"
                 ]
    files = ["UM20", "UM24", "UM25", "UM36", "UM48", "UM50", "UM60", "UM64", "test"]
    for i in range(0, 1):
        start = time.time()    #timer
        hp_seq = hp2d.string_seq_to_bin(sequences[i])
        len_hp_seq = hp_seq.shape[0]
        print(len_hp_seq)
        bounds = [(lower_bound, upper_bound)]*(len_hp_seq-2)
#        result = differential_evolution(hp2d.func_2D_protein, bounds, (hp_seq, settings), strategy='best1bin', disp=True, popsize=len_hp_seq*20, polish=False, workers=1, maxiter=20)
#        result = diff_evol(hp2d.func_2D_protein, bounds, hp_seq,settings, popsize=len_hp_seq*25, maxiter=300, sobol=True)
#        result = diff_evol_max(hp2d.F_2DHP, bounds, hp_seq,settings, popsize=len_hp_seq*25, maxiter=300, sobol=True)
        spiral_settings = {"S":transformation_matrix, "R":mat_R_ij, "r":0.95, "m":250, "theta":45, "kmax":250}
        DE_settings = {'mut':0.8, 'crossp':0.7, 'popsize':len_hp_seq*25, 'maxiter':100}
        cluster_settings = {'m_cluster':250, 'gamma':0.2, 'epsilon':0.6, 'delta':1e-1, 'k_cluster':20}
        result = cluster_DE(hp2d.F_2DHP, bounds, spiral_settings, DE_settings, hp_seq, settings, 
                            m_cluster=cluster_settings['m_cluster'], gamma=cluster_settings['gamma'], 
                            epsilon=cluster_settings['epsilon'], delta=cluster_settings['delta'], k_cluster=cluster_settings['k_cluster'])
        print(result)
        coors = np.array([hp2d.to_2D_cartesian_from_real(res_, settings) for res_ in result])
        Fs = np.array([hp2d.func_2D_protein(res_, hp_seq, settings) for res_ in result])
        print("coor = ", coors)
        end = time.time()
        elapsed_time = end-start
        print("elapsed time = ", end-start)
        data = {'hp_seq':hp_seq, 'x': result, 'coor':coors, 'F':Fs, 'time':elapsed_time, 'params':[settings, spiral_settings, DE_settings, cluster_settings]}
        with open("data/"+files[i]+"_pkl", 'wb') as handle:
            pickle.dump(data, handle)
        with open("data/"+files[i]+"_pkl", 'rb') as handle:
            b = pickle.load(handle)
        print(b)
#        ret_val = np.array(hp2d.return_val)
#        np.save("data/"+files[i]+"_data", hp2d.return_val, allow_pickle=True)
#        ret_v = np.load("data/"+files[i]+"_data.npy", allow_pickle=True)
#        print(ret_v, ret_v.shape)

#    hp_seq = hp2d.string_seq_to_bin("HPPHPH")
#    print("coor = ", hp2d.to_2D_cartesian_from_real(np.array([ 7.99533929,7.29496199,-0.58728589,6.26379575]), settings))
#    print(hp2d.func_2D_protein(np.array([ 7.99533929,7.29496199,-0.58728589,6.26379575]), hp_seq, settings))