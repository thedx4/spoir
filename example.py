import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
import spoir

d=3
q=1
weights=[1, 0.1]
neighbour_depth = 1

data = pd.read_csv('example_checkin.csv', header=None)      
G = nx.from_pandas_edgelist(data-1,0,1)    
data = np.array(data) - 1
N = np.max(np.max(data)) + 1
A = csr_matrix((np.ones(data.shape[0]), (data[:,0],data[:,1])), shape = (N,N))
A += A.T

U_list = spoir.Projection(A, N, d, q)
U = spoir.InitEmbeddings(A, d, weights, U_list = U_list)

 #UPDATE graph
G.add_edge(3,4)
delta_A = csr_matrix((np.ones(1),([3],[4])), shape = (5,5))
delta_A+=delta_A.T

#UPDATE EMBEDDINGS
U2 = spoir.DynUpdate(G, U, A, delta_A, [3,4], U_list, weights, neighbour_depth)
A += delta_A #Update adj matrix