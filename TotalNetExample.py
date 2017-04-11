from SparseNALG import *
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.sparse import coo_matrix
import ToHyperNEATGenome
import ToHyperNEATSubstrate

sets = OrderedDict()
sets['In'] = 9
sets['F1'] = 16
sets['F2'] = 16
sets['A1'] = 4
sets['A2'] = 4
sets['B1'] = 4
sets['B2'] = 4
sets['C'] = 16
sets['Out'] = 7

R1 = np.array(
     [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
R2 = np.array(R1)
R3 = np.array(
     [[1, 0, 0, 0],
      [1, 0, 0, 0],
      [1, 0, 0, 0],
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 1, 0],
      [0, 0, 1, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [0, 0, 0, 1],
      [0, 0, 0, 1],
      [0, 0, 0, 1]])
R5 = np.array(R3)
R4 = np.ones(shape=(sets['A1'], sets['A2']))
R6 = np.ones(shape=(sets['B1'], sets['B2']))
R7 = np.ones(shape=(sets['A2'], sets['C']))
R8 = np.ones(shape=(sets['B2'], sets['C']))
R9 = np.ones(shape=(sets['C'], sets['Out']))

submatrix_map = OrderedDict()
submatrix_map[('In','F1')] = R1
submatrix_map[('In','F2')] = R2
submatrix_map[('F1','A1')] = R3
submatrix_map[('A1','A2')] = R4
submatrix_map[('F2','B1')] = R5
submatrix_map[('B1','B2')] = R6
submatrix_map[('A2','C')] = R7
submatrix_map[('B2','C')] = R8
submatrix_map[('C','Out')] = R9

def apply_subst():
    conn_matrix = TNM(sets, submatrix_map)
    # Substitution pattern
    cort_col = np.array([[0,1,1,1,0],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,0]])
    sparse_matrix = coo_matrix(conn_matrix)
    matrix = np.array(conn_matrix)
    for pair in zip(sparse_matrix.row, sparse_matrix.col):
        matrix = subst(matrix, cort_col, pair[0], pair[1])
    return matrix

def eval_example():
    T1 = R1 * iota(sets['F1'], sets['F2']) + R2 * kappa(sets['F1'], sets['F2'])
    T2 = pi(sets['F1'], sets['F2']) * R3 * iota(sets['A1'], sets['B1']) + \
        rho(sets['F1'], sets['F2']) * R5 * kappa(sets['A1'], sets['B1'])
    T3 = pi(sets['A1'], sets['B1']) * R4 * iota(sets['A2'],sets['B2']) + \
        rho(sets['A1'], sets['B1']) * R6 * kappa(sets['A2'], sets['B2'])
    T4 = pi(sets['A2'], sets['B2']) * R7 + rho(sets['A2'], sets['B2']) * R8
    T5 = R9
    return reduce(np.matmul, [T1,T2,T3,T4,T5])

### Running Example ###
total_net = TNM(sets, submatrix_map)
plt.matshow(1-total_net)
plt.gray()
plt.show()

ToHyperNEATGenome.matrixToGenomeXML(total_net, 'TotalNet.xml')
ToHyperNEATSubstrate.matrixToSubstrate(total_net, 'TotalNetSubstrate.xml')
save_edge_list_csv(total_net, 'TotalNetGraph.csv')

total_net_product = eval_example()
print total_net_product.shape
