import numpy as np
import networkx as nx
import time
import copy

def InitEmbeddings(A, d, weights, *, U_list = None, seed = 0):
    q = len(weights) - 1
    N = A.shape[0]
    np.random.seed(seed)
    if not U_list:
        U_list = Projection(N, d, q)
    U = np.zeros(U_list[0].shape)
    for i in range(len(weights)):
        U += weights[i] * U_list[i]
    return U
    
def Projection(A, N, d, q):
    U_list = [np.random.normal(0, 1.0/np.sqrt(d),(N,d))]
    for i in range(q):
        U_list.append(A.dot(U_list[-1]))
    return U_list
    
def DynUpdate(G, U, A, delta_A, a, U_list, weights, depth = 2, seed = 0):
    q = len(U_list)
    N, d = U_list[0].shape
    N_new = delta_A.shape[0]        
    
    if N_new > N:
        A.resize((N_new,N_new))
        for i in range(1,len(U_list)):
            U_list[i] = np.insert(U_list[i], N, np.zeros((N_new - N, d)),0)
        np.random.seed(seed)                                
        U_list[0] = np.insert(U_list[0], N, np.random.normal(0, 1.0/np.sqrt(d),(N_new - N, d)), 0)
        N = N_new;
    
    a_n=dict()
    
    for b in a:
        b_n=nx.single_source_shortest_path_length(G,b,depth)
        a_n.update(b_n)
    
    sarr=list(a_n.keys())
    delta_U = [np.zeros((len(sarr),d))]
    for i in range(1, q):
        delta_U.append(delta_A[sarr][:,sarr].dot(U_list[i-1][sarr]) + A[sarr][:,sarr].dot(delta_U[i-1]) + delta_A[sarr][:,sarr].dot(delta_U[i-1]))
    for i in range(1, q):
        U_list[i][sarr] += delta_U[i]
    U2 = copy.deepcopy(U)
    U2[sarr] = np.zeros([len(sarr),d])
    for i in range(len(weights)):
        U2[sarr] += weights[i] * U_list[i][sarr]
        
    return U2