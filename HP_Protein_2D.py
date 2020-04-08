# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:37:33 2020

@author: Saint8312
"""
import time
import numpy as np

def func_2D_protein(x, *args):
    return None

def calculate_fitness_2D_sequence(hp_seq, move_seq, settings):
    coord = to_2D_cartesian(move_seq)
    print(coord)
    return fitness

def to_2D_cartesian(move_seq):
    '''
    i assume that F is always +1 on X-axis for the first sequence
    F means 2*prev_coor-prev(prev_coor)
    R means clockwise_rotation(F) (-90 deg)
    L means counter_clockwise_rotation(F) (90 deg)
    '''
    
    len_move_seq = move_seq.shape[0]
    coord = np.zeros((len_move_seq+2, 2))
    coord[0] = np.array([0,0])
    coord[1] = np.array([1,0]) #the first movement is always F
    
    for i in range(len_move_seq):
        vec_diff = coord[i+1]-coord[i]
        if move_seq[i] == 0: #if F
            coord[i+2] = coord[i+1] + vec_diff
        elif move_seq[i] == 1: #if R
            coord[i+2] = coord[i+1] + np.array( [f_r_x(vec_diff[0], vec_diff[1], np.pi/2), f_r_y(vec_diff[0], vec_diff[1], np.pi/2)], dtype=int )
        elif move_seq[i] == 2: #if L
            coord[i+2] = coord[i+1] + np.array( [f_r_x(vec_diff[0], vec_diff[1], -np.pi/2), f_r_y(vec_diff[0], vec_diff[1], -np.pi/2)], dtype=int )
    return coord

f_r_x = lambda x,y,t:(x*np.cos(t))+(y*np.sin(t)) #rotation function for x
f_r_y = lambda x,y,t:(-x*np.sin(t))+(y*np.cos(t)) #rotation function for y

if __name__ == "__main__":

    '''
    2D relative contact
    H=1, P=0
    F=0, R=1, L=2
    contact value HH=-1, HP=0, PH=0, PP=0
    four directional neighboor checking, range : F=[0,3), R=[3,6), L=[6,9)
    the first two seqs always belong to F, default sequence is F^N, N \in Z^+, meaning if a protein seq is [HPPHPH] then the default move seq is [FFFF] or [0000]
    '''
    #inputs
    np.random.seed(13)
    settings={
            "E_contact": {"HH":-1., "HP":0., "PH":0., "PP":0.},
            "ranges":np.array(((0,3),(3,6),(6,9)))
            }
    hp_seq = [1,0,0,1,0,1]
    len_hp_seq = len(hp_seq)
    x = np.random.uniform(0,9,len_hp_seq-2)
    print(x)
    
    
    
    #range to sequence
    ranges = settings["ranges"]
    len_x = x.shape[0]
    move_seq = np.empty(len_x)
    for i in range(len_x):
        if ranges[0][0]<=x[i]<ranges[0][1]:
            move_seq[i] = 0
        elif ranges[1][0]<=x[i]<ranges[1][1]:
            move_seq[i] = 1
        elif ranges[2][0]<=x[i]<ranges[2][1]:
            move_seq[i] = 2
    print(move_seq)
    
    #feasible conformation checker
    #move_seq = np.array([0,0,1,1,2,2,1,0,0,0,1,1,2,2,1,1,2,1,2,2])
    len_move = move_seq.shape[0]
    feasibility = True    
    fitness = np.inf

    counter=1
    for i in range(len_move):
        print(move_seq[i], counter)
        if counter>2:
            print("infeasible")
            feasibility = False
            break
        if (i<len_move-1) and (move_seq[i]>0): #if not end of move_seq and moveement is not F
            current_move = move_seq[i]
            next_move = move_seq[i+1]
            if next_move!=current_move: #check if the next move = current move
                counter=1
                continue
            elif next_move==current_move:
                counter+=1
                continue
    
    #fitness value setter
    #calculate the actual fitness value
    if feasibility == True:
        to_2D_cartesian(move_seq)
        fitness = calculate_fitness_2D_sequence(hp_seq, move_seq, settings)
    
