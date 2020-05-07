# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:37:33 2020

@author: Saint8312
"""
import time
import numpy as np

#global return_val
#return_val = []

f_2DHP = [lambda x, *args : func_2D_protein(x, *args)]
#F_2DHP = lambda x, *args : 1/( 1 + sum([abs(f_(x, *args)) for f_ in f_2DHP]) ) #transform into maximization function, for clustering purpose (doesnt work for protein function since the minimum is not 0)
F_2DHP = lambda x, *args : 1 - (1 / (1 + sum([abs(f_(x, *args)) for f_ in f_2DHP])))

def func_2D_protein(x, *args):
    '''
    2D relative contact
    H=1, P=0
    F=0, R=1, L=2
    contact value HH=-1, HP=0, PH=0, PP=0
    four directional neighboor checking, range : F=[0,3), R=[3,6), L=[6,9)
    the first two seqs always belong to F, default sequence is F^N, N \in Z^+, meaning if a protein seq is [HPPHPH] then the default move seq is [FFFF] or [0000]
    '''
    '''
    x: input of 1D array with size N-2, x \in R, N \in Z^+ (numpy array)
    *args:
        0 = hp_seq (1D array) with size N
        1 = settings:{"E_contact":{"HH", "HP", "PH", "PP"} (\in R), "ranges" ((3,2) \in R)}
    '''
    #inputs
    hp_seq = args[0]
    settings = args[1]
    
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
    
    #feasible conformation checker
    #move_seq = np.array([0,0,1,1,2,2,1,0,0,0,1,1,2,2,1,1,2,1,2,2])
    len_move = move_seq.shape[0]
    feasibility = True    
#    fitness = np.inf #set fitness value as infinite for infeasible conformations (ONLY WORKS FOR MINIMIZATION ALGORITHM due to how the absolute value of objective function is calculated)
    fitness = 0 #set fitness value as 0 for infeasible conformations
    
    #initial feasibility check, to make the calculation faster
    counter=1
    for i in range(len_move):
        if counter>2:
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
        coords = to_2D_cartesian(move_seq)
        if len(np.unique(coords, axis=0)) == len(coords): #another feasibility check by checking duplicate coordinates
            fitness = calculate_fitness_2D_sequence(hp_seq, coords, settings)
            
            '''
            #append to return val to save the data to pickle
            length = len(return_val)
            if length > 0:
                if not np.array_equal(coords, return_val[length-1][0]):
                    return_val.append((coords, fitness))
            else:
                return_val.append((coords, fitness))
            '''
    return fitness


def calculate_fitness_2D_sequence(hp_seq, coords, settings):
    directional_coords = np.array([[1,0],[0,-1],[-1,0],[0,1]]) #clockwise starting from East
    coords_length=coords.shape[0]
    dir_coords_length = directional_coords.shape[0]
    contacts = [] #(p_idx_1, p_idx_2, contact_type)
    #check 4 directional neighboors, check the contact type
    for i in range(coords_length):
        neighboor_coords = coords[i]+directional_coords
        indirect_conn = None
        for j in range (dir_coords_length):
            if i==0: #start index
                indirect_conn = coords[i+2:]
            elif i==dir_coords_length-1: #end index
                indirect_conn = coords[:i-1]
            else :
                indirect_conn = np.concatenate((coords[:i-1], coords[i+2:])) #intermediate index
            idx = np.where((indirect_conn[:,0] == neighboor_coords[j][0]) & (indirect_conn[:,1] == neighboor_coords[j][1]))[0] #check the index of the non direct neighhbors which is in the coords 
            if idx.size>0:
                matched_coord = indirect_conn[idx[0]]
                matched_idx = np.where((coords[:,0] == matched_coord[0]) & (coords[:,1] == matched_coord[1]))[0] #get the index in coords array
                contacts.append((i,matched_idx[0]))
            #print(indirect_conn, neighboor_coords[j])
    if len(contacts)>0:
        contacts = np.unique(np.sort(contacts, axis=1), axis=0)
    
    #calculate the fitness
    E_dict = {1:"H", 0:"P"}
    fitness = np.sum([settings["E_contact"][E_dict[hp_seq[con_[0]]]+E_dict[hp_seq[con_[1]]]] for con_ in contacts])
            
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
    coord[0:2] = np.array([[0,0],[1,0]]) #the first movement is always F
    for i in range(len_move_seq):
        vec_diff = coord[i+1]-coord[i]
        if move_seq[i] == 0: #if F
            coord[i+2] = coord[i+1] + vec_diff
        elif move_seq[i] == 1: #if R
            coord[i+2] = coord[i+1] + np.array( [f_r_x(vec_diff[0], vec_diff[1], np.pi/2), f_r_y(vec_diff[0], vec_diff[1], np.pi/2)], dtype=int )
        elif move_seq[i] == 2: #if L
            coord[i+2] = coord[i+1] + np.array( [f_r_x(vec_diff[0], vec_diff[1], -np.pi/2), f_r_y(vec_diff[0], vec_diff[1], -np.pi/2)], dtype=int )
    return coord

def to_2D_cartesian_from_real(x, settings):
    '''
    the input is directly the x vector
    '''
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
    
    len_move_seq = move_seq.shape[0]
    coord = np.zeros((len_move_seq+2, 2))
    coord[0:2] = np.array([[0,0],[1,0]]) #the first movement is always F
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

def string_seq_to_bin(string_seq):
    '''
    "HHPH" -> [1,1,0,1]
    '''
    dic_seq={"H":1, "P":0}
    return np.array([dic_seq[s_] for s_ in string_seq])
    
    
if __name__ == "__main__":
    np.random.seed(13)
    #inputs
    settings={
            "E_contact": {"HH":-1., "HP":0., "PH":0., "PP":0.},
            "ranges":np.array(((0,3),(3,6),(6,9)))
            }
#    hp_seq = string_seq_to_bin("HPPHPH")
#    len_hp_seq = hp_seq.shape[0]
#    x = np.random.uniform(0,9,len_hp_seq-2) #x should be generated from optimization algorithms
#    print(func_2D_protein(x, hp_seq, settings))
    
    
    
