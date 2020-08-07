# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:55:08 2020

@author: Saint8312
"""
import numpy as np
from scipy.optimize import differential_evolution, rosen
import time
from itertools import repeat
import sobol_seq


def diff_evol(fobj, bounds, *args, mut=0.8, crossp=0.7, popsize=20, maxiter=10, sobol=True, return_all=False):
    dimensions = len(bounds)
    x_arrays = np.zeros((maxiter, dimensions)) #if wanted to return the arrays per iteration
    fitness_arrays = np.zeros((maxiter, 1)) #if wanted to return the fitness value per iteration
    pop = None
    if sobol: #using sobol
        pop = sobol_seq.i4_sobol_generate(dimensions, popsize)
    else: #using random
        pop = np.random.rand(popsize, dimensions) 
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind, *args) for ind in pop_denorm])
    nfev = popsize #increment by the number of population
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(maxiter):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm, *args)
            nfev += 1 #increment for each func eval
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
#        yield best, fitness[best_idx] #default
        x_arrays[i] = best #set current iter best vector
        fitness_arrays[i] = fitness[best_idx] #set current best fitness
#        print("iter=",i,", best fit=",fitness[best_idx])
    if return_all:
        return x_arrays, fitness_arrays, nfev #return the domain vectors, fitness vectors, and total of function evaluation
    else:
        return best, fitness[best_idx]

def diff_evol_max(fobj, bounds, *args, mut=0.8, crossp=0.7, popsize=20, maxiter=10, sobol=True, return_all=False, cauchy_F_tol=1e-7, cauchy_x_tol=1e-7, cauchy_max_counter=50):
    '''
    DE for maximization purpose
    '''
    dimensions = len(bounds)
    x_arrays = np.zeros((maxiter, dimensions)) #if wanted to return the arrays per iteration
    fitness_arrays = np.zeros((maxiter, 1)) #if wanted to return the fitness value per iteration
    pop = None
    if sobol: #using sobol
        pop = sobol_seq.i4_sobol_generate(dimensions, popsize)
    else: #using random
        pop = np.random.rand(popsize, dimensions) 
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind, *args) for ind in pop_denorm])
    nfev = popsize #increment by the number of population
    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]
    itercounter = 0
    counter = 0 #cauchy's parameter
    for i in range(maxiter):
        #cauchy early stopping check
        if counter>cauchy_max_counter:
            break
        x_prev_best = best #cauchy's parameter
        f_prev_best = fitness[best_idx] #cauchy's parameter
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm, *args)
            nfev += 1 #increment for each func eval
            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
#        yield best, fitness[best_idx] #default
        x_arrays[i] = best #set current iter best vector
        fitness_arrays[i] = fitness[best_idx] #set current best fitness
#        print("iter=",i,", best fit=",fitness[best_idx], best)
        #cauchy parameters calculation
        if ( np.fabs(f_prev_best-fitness[best_idx])<cauchy_F_tol ) and ( np.linalg.norm(best-x_prev_best)<cauchy_x_tol ):
            counter+=1
        else:
            counter=0
        itercounter+=1
#    print("maxiter = ",itercounter)
    if return_all:
        return x_arrays, fitness_arrays, nfev #return the domain vectors, fitness vectors, and total of function evaluation
    else:
        print(best, fitness[best_idx], fobj(best, *args))
#        return best, fitness[best_idx]
        return best, fobj(best, *args)




if __name__ == "__main__":

#    init_pop=np.array([
#            [0,0,0],
#            [34,54,67],
#            [2,0,1],
#            [2,1,4],
#            [0,1,0]
#            ])
#    
    
#    start = time.time()
#    bounds = [(0,100), (0,100), (0,100)]
#    result = differential_evolution(F, bounds,(0,100), maxiter=100, strategy='best1bin', disp=False, popsize=100, polish=False)
#    end = time.time()
#    print(result, end-start)
#    
#    
#    start = time.time()
#    bounds = [(0,100), (0,100), (0,100)]
#    result = diff_evol(F,bounds, 0,100, maxiter=100, popsize=100)
##    print(np.array(list(result)))
#    print(result[0])
#    end = time.time()
#    print(end-start)
    

    def F(x,*args):
        print(*args)
        val=0
        for x_ in x:
            val += x_**2
        if val<=args[0]:
            val= -100
        elif val>args[1]:
            val= 100
        return val  
    
    def passf_2(F, x, *args):
        return map(lambda x_: F(x_, *args), [x_ for x_ in x])
    
    def passf_3(F, x, *args):
        t_F = lambda x_: F(x_, *args)
        return map(t_F, x)
    
    def passf(F, x, *args):
#        return map(F, x, args)
        return [F(x_, *args) for x_ in x]        
#    res = map(F, x, [0]*3,[1]*3)
#    print(list(passf(x, 0,1)))
    x = np.array([[0,1,2],[10,2,0],[0,0,0]])
    print(list(passf_3(F, x, 1,100)))
    