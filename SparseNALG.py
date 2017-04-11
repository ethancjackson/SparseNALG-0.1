import numpy as np
import scipy
from scipy.sparse import identity, coo_matrix, csc_matrix

# Returns the coordinates of non-zero (abs_val > epsilon) elements.
def nonzero(A, epsilon=0.0001):
    A = coo_matrix(A)
    nz_mask = (abs(A.data) > epsilon)
    return (A.row[nz_mask],A.col[nz_mask])

# A --> A + B
def iota(sizeA, sizeB):
    assert sizeB >= sizeA, 'Target dimension cardinality can not be smaller than source.'
    A = coo_matrix(identity(sizeA))
    return csc_matrix(coo_matrix((A.data, (A.row,A.col)), shape=(sizeA,sizeB)))

# B --> A + B
def kappa(sizeA, sizeB):
    assert sizeB >= sizeA, 'Target dimension cardinality can not be smaller than source.'
    B = coo_matrix(identity(sizeA))
    return csc_matrix((B.data, (B.row,(sizeB-sizeA+B.col))), shape = (sizeA,sizeB))

# A + B --> A
def pi(sizeA, sizeB):
    return iota(sizeB,sizeA).transpose()

# A + B --> B
def rho(sizeA, sizeB):
    return kappa(sizeB,sizeA).transpose()

# iTL
def inject_top_left(r, xsrc, xtrg):
    return ((iota(r.shape[0], r.shape[0]+xsrc)).transpose() * ((r * iota(r.shape[1], r.shape[1]+xtrg))))

# iTR
def inject_top_right(r, xsrc, xtrg):
    return ((iota(r.shape[0], r.shape[0]+xsrc)).transpose() * ((r * kappa(r.shape[1], r.shape[1]+xtrg))))

# iBL
def inject_bot_left(r, xsrc, xtrg):
    return ((kappa(r.shape[0], r.shape[0]+xsrc)).transpose() * ((r * iota(r.shape[1], r.shape[1]+xtrg))))

# iBR
def inject_bot_right(r, xsrc, xtrg):
    return ((kappa(r.shape[0], r.shape[0]+xsrc)).transpose() * ((r * kappa(r.shape[1], r.shape[1]+xtrg))))

# Connect
def connect(ins, p, outs):
    assert ins.shape[1] == p.shape[0]
    assert p.shape[0] == p.shape[1]
    assert p.shape[1] == outs.shape[0]
    A = ins.shape[0]
    N = p.shape[0]
    B = outs.shape[1]
    m1 = inject_top_right(ins, N, A)
    m2 = inject_bot_right(p, A, A)
    m3 = inject_top_left(m1 + m2, B, B)
    m4 = inject_top_right(outs, B, A + N)
    m5 = inject_bot_right(m4, A, 0)
    return m3 + m5

# Substitution
def subst(Net, S, Nx, Ny):
    assert Net.shape[0] == Net.shape[1], \
        'Net must be square.'
    assert S.shape[0] == S.shape[1], \
        'S must be square.'
    assert nonzero(Net[Nx,Ny])
    N = Net.shape[0]
    C = S.shape[0]
    Net[Nx,Ny] = 0
    m1 = inject_top_left(Net,C,C) + inject_bot_right(S,N,N)
    C1 = N
    Cn = N+C-1
    m1[Nx,C1] = 1
    m1[Cn,Ny] = 1
    return m1

# Total Network Matrix
def TNM(S,M):
    S_relsum = np.sum(S.values())
    matrix = csc_matrix((S_relsum,S_relsum))
    for (src, trg) in M:
        x_idx = list(S).index(src)
        y_idx = list(S).index(trg)
        x_extra = np.sum(S.values()[x_idx + 1:])
        y_extra = np.sum(S.values()[y_idx + 1:])
        R = M[(src, trg)]
        x = S[src]
        y = S[trg]
        r = rho(S_relsum, x + x_extra)
        p = pi(x + x_extra, x)
        i = iota(y, y + y_extra)
        k = kappa(y + y_extra, S_relsum)
        matrix += r*p*R*i*k
    return matrix

### Other Functions ###

# Combine two connectivity matrices into one with non-overlapping neurons
def comb_matrix(A, B):
    Q1 = (pi(A.shape[0] + B.shape[0], A.shape[0]) * (A * iota(A.shape[1], A.shape[1] + B.shape[1])))
    Q2 = (rho(A.shape[0] + B.shape[0], B.shape[0]) * (B * kappa(B.shape[1], A.shape[1] + B.shape[1])))
    return Q1 + Q2

# Random Matrix with parametric density
def rnd_matrix(sizeA, sizeB, density):
    return scipy.sparse.rand(sizeA, sizeB, density=density)

# Random relation with parametric density
def rnd_relation(sizeA, sizeB, density):
    return csc_matrix(np.core.ceil(rnd_matrix(sizeA,sizeB,density)))

# Randomize the weights of a connectivity matrix accoring to normal distribution
def randomize_weights_normal(A, mean=0, stdev=0.1):
    return csc_matrix(A * np.random.normal(mean,stdev,A.shape))

# Save connectivity matrix as an unweighted graph (edge list CSV)
def save_edge_list_csv(matrix, filename):
    assert matrix.shape[0] == matrix.shape[1], 'Matrix must be square.'
    A = coo_matrix(matrix, dtype='int')
    edge_list = np.array(list(zip(nonzero(A)[0].tolist(), nonzero(A)[1].tolist())), dtype='int')
    np.savetxt(filename, edge_list, delimiter=',', fmt='%i')
    f = open(filename, 'r+')
    content = f.read()
    f.seek(0,0)
    f.write('Source,Target' + '\n' + content)

# Save connectivity matrix as a weighted graph (edge list CSV)
def save_weighted_edge_list_csv(matrix, filename):
    assert matrix.shape[0] == matrix.shape[1], 'Matrix must be square.'
    A = np.array(matrix)
    edge_list = np.array(list(zip(nonzero(A)[0].tolist(), nonzero(A)[1].tolist())), dtype='int')
    weight_list = []
    for pair in edge_list:
        weight_list.append([A[pair[0],pair[1]]])
    weight_list = np.array(weight_list)
    np.savetxt(filename, np.hstack((edge_list,weight_list)), delimiter=',')
    f = open(filename, 'r+')
    content = f.read()
    f.seek(0,0)
    f.write('Source,Target,Weight' + '\n' + content)
