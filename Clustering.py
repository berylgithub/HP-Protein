# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:54:02 2020

@author: Saint8312
"""
import numpy as np
import time
import sobol_seq
from differential_evolution import diff_evol as DE, diff_evol_max as DE_max


'''
===============================
modified clustering functions: 
===============================
'''


def cluster_DE_mm(F, domain, spiral_settings, DE_settings, *f_args, m_cluster=10, epsilon=1e-5, delta=1e-2, k_cluster=10):
    '''
    modified clustering method, replaced spiral with differential evolution on intensification phase, for multimodal optimization
    DE_settings:
        'mut': mutation prob
        'crossp': crossover prob
        'popsize': population size 
        'maxiter': maximum iteration
    '''
    cluster = clustering_mm(F, domain, spiral_settings,*f_args, 
                            m_cluster=m_cluster, epsilon=epsilon, delta=delta, k_cluster=k_cluster)
    print(cluster)
    dim = len(domain)
    num_cluster = len(cluster["center"])
    new_domains = np.zeros((num_cluster, dim, 2))
    accepted_roots = []
    accepted_fs=[]
    for i in range(num_cluster): #do spiral for each clusters
        print("cluster no. ",i)
        for j in range(dim):
            new_domains[i][j] = np.array([cluster["center"][i][j]-cluster["radius"][i], cluster["center"][i][j]+cluster["radius"][i]])
        x_star, f_star = DE_max(F, new_domains[i], *f_args, mut=DE_settings['mut'], crossp=DE_settings['crossp'], 
                            popsize=DE_settings['popsize'], maxiter=DE_settings['maxiter'])
        if (F(x_star-epsilon, *f_args)<f_star) and (F(x_star+epsilon, *f_args)<f_star): #roots selection by threshold
            accepted_roots.append(x_star)
            accepted_fs.append(f_star)
    #selection by proximity:
    accepted_roots = np.array(accepted_roots)
    print(accepted_roots, len(accepted_roots))
    accepted_fs = np.array(accepted_fs)
    length = accepted_roots.shape[0]
    geq_delta_idxes = []
    leq_delta_idxes = []
    for i in range(length):
        geq_truth = True
        for j in range(length):
            if i!=j:    
                if np.linalg.norm(accepted_roots[i]-accepted_roots[j]) <= delta:
                    geq_truth = False
        if geq_truth:
            geq_delta_idxes.append(i)
        else:
            leq_delta_idxes.append(i)
            
    new_accepted_roots = []
    if len(leq_delta_idxes)>0:
        leq_delta_idxes = np.array(leq_delta_idxes)
        max_root = accepted_roots[np.argmax(accepted_fs[leq_delta_idxes])]  #get the roots with highest F from less than delta idxes
        new_accepted_roots.append(max_root)
    if len(geq_delta_idxes)>0:
        geq_delta_idxes = np.array(geq_delta_idxes)
        new_accepted_roots.extend(accepted_roots[geq_delta_idxes]) #get the greater than delta x-idxes
    new_accepted_roots = np.array(new_accepted_roots)
    
    print(geq_delta_idxes, leq_delta_idxes)
    print(new_accepted_roots, len(new_accepted_roots))
    
    return new_accepted_roots 
       
def clustering_mm(F, domain, spiral_settings, *f_args, m_cluster=10, epsilon=1e-5, delta=1e-2, k_cluster=10):
    '''
    diversification phase of initial guess points, outputs clusters of domains, for multimodal problem
    cluster_settings contains:
        S: transformation matrix
        R: Rij entries
        r: contraction constant
        theta: rotation constant
    '''

    ############## settings for points' rotation
    S = spiral_settings["S"]
    R = spiral_settings["R"]
    r = spiral_settings["r"]
    theta = spiral_settings["theta"]
#    kmax = spiral_settings["kmax"]
    ##############
    dim = len(domain)
    x = sobol_seq.i4_sobol_generate(dim, m_cluster)
    for i in range(dim):
        x.T[i] = x.T[i]*(domain[i][1]-domain[i][0])+domain[i][0]
#    x = np.array([[np.random.uniform(domain[j][0], domain[j][1]) for j in range(dim)] for i in range(m_cluster)]) #generate init population
    temp_F = lambda x_ : F(x_, *f_args)
    center_idx = np.argmax(np.array(list(map(temp_F, x)))) #get the center of cluster using map
#    center_idx = np.argmax(np.array(list(map(F, x)))) #get the center of cluster index
    x_star = x[center_idx] #center of cluster
    radius = np.min(np.array([np.fabs(dom[1]-dom[0]) for dom in domain]))/2.0 #get the radius of cluster
    cluster = {"center": [x_star], "id":[center_idx], "radius":[radius]} #cluster data structure
    #should be another loop here
    for k in range(k_cluster):
        print("k-cluster = ", k)
        for i in range(m_cluster):
            if (i not in cluster["id"]):
                cluster = cluster_f(F, domain, x[i], cluster, i, *f_args)
            x_p = x[np.argmax(np.array(list(map(temp_F, x))))]
#            x_p = x[np.argmax(np.array(list(map(F, x))))]
            x = np.array([rotate_point(x[i], x_p, S, R, dim, r, theta) for i in range(len(x))])
    return cluster

def cluster_DE(F, domain, spiral_settings, DE_settings, *f_args, m_cluster=10, gamma=0.2, epsilon=1e-5, delta=1e-2, k_cluster=10):
    '''
    modified clustering method, replaced spiral with differential evolution on intensification phase
    DE_settings:
        'mut': mutation prob
        'crossp': crossover prob
        'popsize': population size 
        'maxiter': maximum iteration
    '''
    cluster = clustering(F, domain, spiral_settings,*f_args, 
                            m_cluster=m_cluster, gamma=gamma, epsilon=epsilon, delta=delta, k_cluster=k_cluster)
    print(cluster)
    dim = len(domain)
    num_cluster = len(cluster["center"])
    new_domains = np.zeros((num_cluster, dim, 2))
    accepted_roots = []
    accepted_fs=[]
    for i in range(num_cluster): #do spiral for each clusters
        print("cluster no. ",i)
        for j in range(dim):
            new_domains[i][j] = np.array([cluster["center"][i][j]-cluster["radius"][i], cluster["center"][i][j]+cluster["radius"][i]])
        x_star, f_star = DE_max(F, new_domains[i], *f_args, mut=DE_settings['mut'], crossp=DE_settings['crossp'], 
                            popsize=DE_settings['popsize'], maxiter=DE_settings['maxiter'])
        if (1.0-f_star) < epsilon: #roots selection by threshold
            accepted_roots.append(x_star)
            accepted_fs.append(f_star)
    #selection by proximity:
    accepted_roots = np.array(accepted_roots)
    print(accepted_roots, len(accepted_roots))
    accepted_fs = np.array(accepted_fs)
    length = accepted_roots.shape[0]
    geq_delta_idxes = []
    leq_delta_idxes = []
    for i in range(length):
        geq_truth = True
        for j in range(length):
            if i!=j:    
                if np.linalg.norm(accepted_roots[i]-accepted_roots[j]) <= delta:
                    geq_truth = False
        if geq_truth:
            geq_delta_idxes.append(i)
        else:
            leq_delta_idxes.append(i)
            
    new_accepted_roots = []
    if len(leq_delta_idxes)>0:
        leq_delta_idxes = np.array(leq_delta_idxes)
        max_root = accepted_roots[np.argmax(accepted_fs[leq_delta_idxes])]  #get the roots with highest F from less than delta idxes
        new_accepted_roots.append(max_root)
    if len(geq_delta_idxes)>0:
        geq_delta_idxes = np.array(geq_delta_idxes)
        new_accepted_roots.extend(accepted_roots[geq_delta_idxes]) #get the greater than delta x-idxes
    new_accepted_roots = np.array(new_accepted_roots)
    
    print(geq_delta_idxes, leq_delta_idxes)
    print(new_accepted_roots, len(new_accepted_roots))
    
    return new_accepted_roots        
        
'''
=======================================================
========= default clustering functions (Sidarto, 2015) :
=======================================================
'''
def cluster_spiral(F, domain, spiral_settings, *f_args, m_cluster=10, gamma=0.2, epsilon=1e-5, delta=1e-2, k_cluster=10):
    '''
    combination of clustering and spiral optimization
    '''

    cluster = clustering(F, domain, spiral_settings,*f_args, 
                            m_cluster=m_cluster, gamma=gamma, epsilon=epsilon, delta=delta, k_cluster=k_cluster)
    print(cluster)
    dim = len(domain)
    num_cluster = len(cluster["center"])
    new_domains = np.zeros((num_cluster, dim, 2))
    accepted_roots = []
    accepted_fs=[]
    for i in range(num_cluster): #do spiral for each clusters
        print("cluster no. ",i)
        for j in range(dim):
            new_domains[i][j] = np.array([cluster["center"][i][j]-cluster["radius"][i], cluster["center"][i][j]+cluster["radius"][i]])
        x_star, f_star = spiral_dynamics_optimization(F, spiral_settings["S"], spiral_settings["R"], spiral_settings["m"], 
                                     spiral_settings["theta"], spiral_settings["r"], 
                                     spiral_settings["kmax"], new_domains[i], *f_args, log=False)
        if (1.0-f_star) < epsilon: #roots selection by threshold
            accepted_roots.append(x_star)
            accepted_fs.append(f_star)
    #selection by proximity:
    accepted_roots = np.array(accepted_roots)
    print(accepted_roots, len(accepted_roots))
    accepted_fs = np.array(accepted_fs)
    length = accepted_roots.shape[0]
    geq_delta_idxes = []
    leq_delta_idxes = []
    for i in range(length):
        geq_truth = True
        for j in range(length):
            if i!=j:    
                if np.linalg.norm(accepted_roots[i]-accepted_roots[j]) <= delta:
                    geq_truth = False
        if geq_truth:
            geq_delta_idxes.append(i)
        else:
            leq_delta_idxes.append(i)
            
    new_accepted_roots = []
    if len(leq_delta_idxes)>0:
        leq_delta_idxes = np.array(leq_delta_idxes)
        max_root = accepted_roots[np.argmax(accepted_fs[leq_delta_idxes])]  #get the roots with highest F from less than delta idxes
        new_accepted_roots.append(max_root)
    if len(geq_delta_idxes)>0:
        geq_delta_idxes = np.array(geq_delta_idxes)
        new_accepted_roots.extend(accepted_roots[geq_delta_idxes]) #get the greater than delta x-idxes
    new_accepted_roots = np.array(new_accepted_roots)
    
    print(geq_delta_idxes, leq_delta_idxes)
    print(new_accepted_roots, len(new_accepted_roots))
    
    return new_accepted_roots
    
def clustering(F, domain, spiral_settings, *f_args, m_cluster=10, gamma=0.2, epsilon=1e-5, delta=1e-2, k_cluster=10):
    '''
    diversification phase of initial guess points, outputs clusters of domains
    cluster_settings contains:
        S: transformation matrix
        R: Rij entries
        r: contraction constant
        theta: rotation constant
    '''

    ############## settings for points' rotation
    S = spiral_settings["S"]
    R = spiral_settings["R"]
    r = spiral_settings["r"]
    theta = spiral_settings["theta"]
#    kmax = spiral_settings["kmax"]
    ##############
    dim = len(domain)
    x = sobol_seq.i4_sobol_generate(dim, m_cluster)
    for i in range(dim):
        x.T[i] = x.T[i]*(domain[i][1]-domain[i][0])+domain[i][0]
#    x = np.array([[np.random.uniform(domain[j][0], domain[j][1]) for j in range(dim)] for i in range(m_cluster)]) #generate init population
    temp_F = lambda x_ : F(x_, *f_args)
    center_idx = np.argmax(np.array(list(map(temp_F, x)))) #get the center of cluster using map
#    center_idx = np.argmax(np.array(list(map(F, x)))) #get the center of cluster index
    x_star = x[center_idx] #center of cluster
    radius = np.min(np.array([np.fabs(dom[1]-dom[0]) for dom in domain]))/2.0 #get the radius of cluster
    cluster = {"center": [x_star], "id":[center_idx], "radius":[radius]} #cluster data structure
    #should be another loop here
    for k in range(k_cluster):
        print("k-cluster = ", k)
        for i in range(m_cluster):
            if (F(x[i], *f_args) > gamma) and (i not in cluster["id"]):
                cluster = cluster_f(F, domain, x[i], cluster, i, *f_args)
            x_p = x[np.argmax(np.array(list(map(temp_F, x))))]
#            x_p = x[np.argmax(np.array(list(map(F, x))))]
            x = np.array([rotate_point(x[i], x_p, S, R, dim, r, theta) for i in range(len(x))])
    return cluster
    
def cluster_f(F, domain, y, cluster, y_id, *f_args):
    idx = np.argmin(np.array([np.linalg.norm(y-center) for center in cluster["center"]])) #find closest cluster idx
    x_c = cluster["center"][idx] #the closest cluster center to y
    x_t = 0.5*(x_c+y)
    F_ = lambda x_: F(x_,*f_args)
    if (F_(x_t) < F_(y)) and (F_(x_t) < F_(x_c)):
        cluster["center"].append(y)
        cluster["id"].append(y_id)
        cluster["radius"].append(np.linalg.norm(y-x_t))
    elif (F_(x_t) > F_(y)) and (F_(x_t) > F_(x_c)):
        cluster["center"].append(y)
        cluster["id"].append(y_id)
        cluster["radius"].append(np.linalg.norm(y-x_t))
        cluster_f(F, domain, x_t, cluster, -1, *f_args)
    elif F_(y) > F_(x_c):
        cluster["radius"][idx] = np.linalg.norm(y-x_t)
    return cluster
    
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


def spiral_dynamics_optimization(F, S, R, m, theta, r, kmax, domain, *f_args, log=True):
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

    x = sobol_seq.i4_sobol_generate(x_dim, m) #now using sobol
    for i in range(x_dim):
        x.T[i] = x.T[i]*(domain[i][1]-domain[i][0])+domain[i][0]

    f = np.array([F(x_, *f_args) for x_ in x])

    # search the minimum/maximum (depends on the problem) of f(init_point), in this case, max
    x_star = x[np.argmax(f)]
            
    # rotate the points
    for k in range(kmax):
        x = np.array([rotate_point(x[i], x_star, S, R, x_dim, r, theta) for i in range(len(x))])
        f = np.array([F(x_, *f_args) for x_ in x])
        x_star_next = x[np.argmax(f)]
        if(F(x_star_next) > F(x_star)):
            x_star = np.copy(x_star_next)
        if log:
            print("Iteration\tx_star\tF(x_star)")
            print(str(k)+" "+str(x_star)+" "+str(F(x_star, *f_args)))
    return x_star, F(x_star, *f_args)

rotate_point = lambda x, x_star, S, R, x_dim, r, theta: np.matmul( S(R, x_dim, r, theta), x ) - np.matmul( ( S(R, x_dim, r, theta) - np.identity(x_dim) ), x_star )





if __name__=="__main__":
    # seed the random, as usual, to create reproducible result
    np.random.seed(13)
#    domain=np.array([[-4,4], [-4,4]])
#    f=[lambda x : ( (x[0]**4) - 16*(x[0]**2) + 5*x[0] )/2 + ( x[1]**4 - 16*(x[1]**2) + 5*x[1] )/2]
#    #transform f(x) into maximization function
#    F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
#    x_star, F_x_star = spiral_dynamics_optimization(F, transformation_matrix, mat_R_ij, 20, 30, 0.9, 350, domain, log=True)
#    print("\nFinal value of x, F(x) is : ",x_star,",",F_x_star,"")

#    start = time.time()
#    f = [lambda x : x[0]**2 + x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) -0.4*np.cos(4*np.pi*x[1]) +0.7]
#    F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
#    domain = np.array([[-100,100]]*2)    
#    x_star, F_x_star = spiral_dynamics_optimization(F, transformation_matrix, mat_R_ij, 20, 45, 0.95, 350, domain, log=True)
#    print("\nFinal value of x, F(x) is : ",x_star,",",F_x_star,"")
#    print(time.time()-start)
    
#    start = time.time()
#    S = transformation_matrix(mat_R_ij, 1000, 0.9, 30)
#    print(S)
#    print(time.time()-start)

#Problem 1
#    '''with spiral'''
#    f = [lambda x : np.exp(x[0]-x[1])-np.sin(x[0]+x[1]), lambda x : (x[0]**2) * (x[1]**2) - np.cos(x[0]+x[1])]
#    F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
#    domain = np.array([[-10,10]]*2)  
#    start = time.time()
#    spiral_settings = {"S":transformation_matrix, "R":mat_R_ij, "r":0.95, "m":250, "theta":45, "kmax":250}
#    cluster_spiral(F, domain, spiral_settings, m_cluster=250, gamma=0.2, epsilon=0.1, delta=1e-1, k_cluster=10)
#    print(time.time()-start)
    
#    '''with DE'''
#    f = [lambda x : np.exp(x[0]-x[1])-np.sin(x[0]+x[1]), lambda x : (x[0]**2) * (x[1]**2) - np.cos(x[0]+x[1])]
#    F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
#    domain = np.array([[-10,10]]*2)  
#    start = time.time()
#    spiral_settings = {"S":transformation_matrix, "R":mat_R_ij, "r":0.95, "m":250, "theta":45, "kmax":250}
#    DE_settings = {'mut':0.8, 'crossp':0.7, 'popsize':100, 'maxiter':20}
#    cluster_DE(F, domain, spiral_settings, DE_settings, m_cluster=250, gamma=0.2, epsilon=0.1, delta=1e-1, k_cluster=10)
#    print(time.time()-start)
    
    
#    '''with spiral'''
#    f = [lambda x : 2*x[0]+x[1]+x[2]+x[3]+x[4]-6,
#         lambda x : 2*x[1]+x[0]+x[2]+x[3]+x[4]-6,
#         lambda x : 2*x[2]+x[0]+x[1]+x[3]+x[4]-6,
#         lambda x : 2*x[3]+x[0]+x[2]+x[1]+x[4]-6,
#         lambda x : x[0]*x[1]*x[2]*x[3]*x[4] - 1
#         ]
#    F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
#    domain = np.array([[-10,10]]*5)  
#    start = time.time()
#    spiral_settings = {"S":transformation_matrix, "R":mat_R_ij, "r":0.95, "m":200, "theta":45, "kmax":250}
#    cluster_spiral(F, domain, spiral_settings, m_cluster=1000, gamma=0.1, epsilon=0.1, delta=0.1, k_cluster=10)
#    print(time.time()-start)    
    
#    '''with DE'''
#    f = [lambda x : 2*x[0]+x[1]+x[2]+x[3]+x[4]-6,
#         lambda x : 2*x[1]+x[0]+x[2]+x[3]+x[4]-6,
#         lambda x : 2*x[2]+x[0]+x[1]+x[3]+x[4]-6,
#         lambda x : 2*x[3]+x[0]+x[2]+x[1]+x[4]-6,
#         lambda x : x[0]*x[1]*x[2]*x[3]*x[4] - 1
#         ]
#    F = lambda x : 1/( 1 + sum([abs(f_(x)) for f_ in f]) )
#    domain = np.array([[-10,10]]*5)  
#    start = time.time()
#    DE_settings = {'mut':0.8, 'crossp':0.7, 'popsize':100, 'maxiter':30}
#    spiral_settings = {"S":transformation_matrix, "R":mat_R_ij, "r":0.95, "m":200, "theta":45, "kmax":250}
#    cluster_DE(F, domain, spiral_settings, DE_settings, m_cluster=300, gamma=0.1, epsilon=0.1, delta=0.1, k_cluster=10)
#    print(DE_max(F, domain, mut=DE_settings['mut'], crossp=DE_settings['crossp'], popsize=DE_settings['popsize'], maxiter=DE_settings['maxiter']))
#    print(time.time()-start)    
    
    '''multimodal optimization, only possible with clustering'''
    domain = np.array([[-4,4]]*2)
    F = lambda x : ( (x[0]**4) - 16*(x[0]**2) + 5*x[0] )/2.0 + ( (x[1]**4) - 16*(x[1]**2) + 5*x[1] )/2.0
    G = lambda x : -F(x)
    start = time.time()
    spiral_settings = {"S":transformation_matrix, "R":mat_R_ij, "r":0.95, "m":300, "theta":45, "kmax":200}
    DE_settings = {'mut':0.8, 'crossp':0.7, 'popsize':100, 'maxiter':50}
    results = cluster_DE_mm(G, domain, spiral_settings, DE_settings, m_cluster=300, epsilon=0.1, delta=0.15, k_cluster=10)
    print(list(map(F, results)))
    print(time.time()-start)
    