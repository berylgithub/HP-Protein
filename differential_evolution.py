# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:55:08 2020

@author: Saint8312
"""
import numpy as np
from scipy.optimize import differential_evolution, rosen
import time

def F(x,*args):
    val=0
    for x_ in x:
        val += x_**2
    return f(x, val, args)    
#    if val<=args[0]:
#        val= -100
#    elif val>args[1]:
#        val= 100
#    return val

def f(x, val, args):
    if val<=args[0]:
        val= -100
    elif val>args[1]:
        val= 100
    return val


if __name__ == "__main__":
    '''
    start = time.time()
    bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]
    result = differential_evolution(rosen, bounds, workers=1)
    end = time.time()
    print(result, end-start)
    print(rosen(result.x))
    '''
    init_pop=np.array([
            [0,0,0],
            [34,54,67],
            [2,0,1],
            [2,1,4],
            [0,1,0]
            ])
    
    
    start = time.time()
    bounds = [(0,100), (0,100), (0,100)]
    result = differential_evolution(F, bounds,(0,100), maxiter=10, strategy='best1bin', disp=True, popsize=100)
    end = time.time()
    print(result.x, end-start)
    
    
    
    
    
    
    