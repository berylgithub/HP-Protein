# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:54:02 2020

@author: Saint8312
"""
import numpy as np
import time

def mat_R_ij(dim, i, j, theta):
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    R = np.zeros((dim, dim))
    for a in range(dim):
        for b in range(dim):
            if ((a==i) and (b==i)) or ((a==j) and (b==j)):
                R[a][b] = c
            elif ((a==j) and (b==i)):
                R[a][b] = s
            elif ((a==i) and (b==j)):
                R[a][b] = -s
            else:
                if(a==b):
                    R[a][b] = 1
                else :
                    R[a][b] = 0
    return R


def transformation_matrix(mat_R_ij, dim, r, theta):
    '''
    generate the matrix of S(n) = r(n)*R(n), where r = contraction matrix, R = rotation matrix
    '''
    # generate r matrix
    mat_r = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if i==j:
                mat_r[i][j]=r
                    
    # generate R(n) matrix
    mat_Rn = np.zeros((dim, dim))
#    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    if(dim<2):
        print("Dimension is not viable !!")
    elif(dim>=2):
        R=np.identity(dim)
        for i in range(dim-1):
            for j in range(i):
                R=np.matmul(R, mat_R_ij(dim, i,j, theta))
        mat_Rn = R
    # S(n) = r(n)*R(n)
    return np.matmul(mat_r, mat_Rn)


def spiral_dynamics_optimization(F, S, R, m, theta, r, kmax, domain, log=True):
    '''
    Function F optimization using spiral dynamics -> rotating & contracting points
    '''
    
    x_dim = domain.shape[0]
    
    if log:
        print("Init points = ",m)
        print("Theta = ",theta)
        print("r = ",r)
        print("iter_max = ",kmax)
        print("dimension = ",x_dim)
        print("domain = ",domain)
        print()
    
    # generate m init points using random uniform between function domain
    x = np.array([[np.random.uniform(domain[j][0], domain[j][1]) for j in range(x_dim)] for i in range(m)])

    f = np.array([F(x_) for x_ in x])

    # search the minimum/maximum (depends on the problem) of f(init_point), in this case, max
    x_star = x[np.argmax(f)]
            
    # rotate the points
    for k in range(kmax):
        x = np.array([rotate_point(x[i], x_star, S, R, x_dim, r, theta) for i in range(len(x))])
        f = np.array([F(x_) for x_ in x])
        x_star_next = x[np.argmax(f)]
        if(F(x_star_next) > F(x_star)):
            x_star = np.copy(x_star_next)
        if log:
            print("Iteration\tx_star\tF(x_star)")
            print(str(k)+" "+str(x_star)+" "+str(F(x_star)))
    return x_star, F(x_star)

rotate_point = lambda x, x_star, S, R, x_dim, r, theta: np.matmul( S(R, x_dim, r, theta), x ) - np.matmul( ( S(R, x_dim, r, theta) - np.identity(x_dim) ), x_star )

    




if __name__=="__main__":
    # seed the random, as usual, to create reproducible result
    np.random.seed(13)
#    domain=np.array([[-4,4], [-4,4]])
#    f=[lambda x : ( (x[0]**4) - 16*(x[0]**2) + 5*x[0] )/2 + ( x[1]**4 - 16*(x[1]**2) + 5*x[1] )/2]
#    #transform f(x) into maximization function
#    F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
#    
#    x_star, F_x_star = spiral_dynamics_optimization(F, transformation_matrix, mat_R_ij, 20, 30, 0.9, 350, domain, log=True)
#    print("\nFinal value of x, F(x) is : ",x_star,",",F_x_star,"")

    start = time.time()
    f = [lambda x : x[0]**2 + x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) -0.4*np.cos(4*np.pi*x[1]) +0.7]
    F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
    domain = np.array([[-100,100]]*20)    
    x_star, F_x_star = spiral_dynamics_optimization(F, transformation_matrix, mat_R_ij, 20, 30, 0.9, 350, domain, log=True)
    print("\nFinal value of x, F(x) is : ",x_star,",",F_x_star,"")
    print(start-time.time())
    
#    start = time.time()
#    S = transformation_matrix(mat_R_ij, 1000, 0.9, 30)
#    print(S)
#    print(time.time()-start)
