import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, os

import Cython;
import molmod as mm;
import tensorflow as tf

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_atomic_features(n):
    '''n is the atomic number'''
    return


def add_sample(url,nodes,features,target,A,sizes,molecule_id,elements_info):
    properties = [];
    with open(url,'r') as file:
        for row in file:
            properties += row.split();
    #extract information from xyz file
    mol = mm.Molecule.from_file(url);
    #mol.write_to_file("new.xyz");
    mol.graph = mm.MolecularGraph.from_geometry(mol);
    vertices = mol.graph.numbers;
    edges = mol.graph.edges;
    d = len(vertices) #the size of each molecule
    #np.append(nodes,vertices)
    #nodes = np.append(nodes,vertices)
    
    atomic_number_mean = elements_info[1]
    atomic_number_stdev = elements_info[2]

    dipole_moment = float(properties[6])
    polarizability = float(properties[7])
    homo = float(properties[8])
    lumo = float(properties[9])
    gap = float(properties[10])
    enthalpy = float(properties[15])
    free_nrg = float(properties[16])
    heat_capacity = float(properties[17])
  
    tempA = np.zeros((d,d)); #Adjacency matrix
    #Structure of the features matrix: (in a row)
    # atomic_no, H, C, N, O, F, acceptor, donor, aromatic, hybridization
    # int, one-hot (5 cols), bool, bool, bool, one-hot

    f = 6
    tempfeatures = [[0]*f for _ in range(d)]; # d=#nodes,  f=#features available

    #populate the adjacency matrix with intermolecular distances in terms of 1/r^2
    for tupl in edges:
        tuple_list = list(tupl);
        #print(tuple_list)
        v_i = tuple_list[0];
        v_j = tuple_list[1];
        tempA[v_i][v_j] = 1.0/pow(mol.distance_matrix[v_i][v_j],2)
        tempA[v_j][v_i] = 1.0/pow(mol.distance_matrix[v_i][v_j],2)
        #print(mol.distance_matrix[v_i][v_j],mol.distance_matrix[v_j][v_i])

    for atom in range(len(vertices)):
        tempfeatures[atom][0] = (float(vertices[atom])-atomic_number_mean)/atomic_number_stdev
        tempfeatures[atom][2] = int(vertices[atom]==6) #C
        tempfeatures[atom][3] = int(vertices[atom]==7) #N
        tempfeatures[atom][4] = int(vertices[atom]==8) #O
        tempfeatures[atom][5] = int(vertices[atom]==9) #F
        #tempfeatures[atom][6] = list(vertices).count(1) #number of H

        #print(tempfeatures[v_i][1])
    A.append(tempA)
    if (molecule_id == 0):
        sizes = sizes + [d-1]
    else:
        sizes = sizes + [d]
    molecule_id=molecule_id+1

    target.append([dipole_moment,polarizability,homo,lumo,gap,
                    enthalpy,free_nrg,heat_capacity])
    features+=tempfeatures
    return nodes, features, target, A, sizes, molecule_id

def load_data3():
    """Load data."""
    path="../tem/"
    nodes = np.array([])
    features = [] #features of each node
    A=[] #list of graph adjacency matrices; each entry is the adjacency matrix for one molecule
    sizes = [] #list of sizes of molecules; each entry is the size of a molecule
    molecule_id = 0
    target = [] #list of "y's" - each entry is an "answer" for a molecule

    # Info for standardizing data
    elements = np.array([1,6,7,8,9])
    elements_mean = np.mean(elements)
    elements_stdev = np.std(elements)
    elements_all = [elements, elements_mean, elements_stdev]

    for file in os.listdir(path):
        nodes, features, target, A, sizes, molecule_id = add_sample(path+file,nodes,features,target,A,sizes,molecule_id,elements_all)

    molecule_partitions=np.cumsum(sizes) #to get partition positions
    n = molecule_partitions[-1]+1 #total sum of all nodes
    adj = np.zeros((n,n))



    i=0 #index
    j=0
    v=0
    t=0
    for matrix in A:
        #put the list of matrices into one big matrix
        size = len(matrix)
        adj[i:(i+size),i:(i+size)] = matrix
        i += size
        j += 1
        if (j == 600):
            v = i
            #where validation set begins
        if (j == 900):
            #where the test set begins
            t = i

    labels = np.array(target)

    sparse_adj = sp.csr_matrix(adj);

    #idx_test = range(t,n)
    #idx_train = range(0,v-1)
    #idx_val = range(v,t)
    idx_train = range(600)
    idx_val = range(600,900)
    idx_test = range(900,1000)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_test[test_mask] = labels[test_mask]
    y_val[val_mask] = labels[val_mask]

    feats = sp.coo_matrix(np.array(features)).tolil()
    return sparse_adj, feats, y_train, y_val, y_test, train_mask, val_mask, test_mask, molecule_partitions, molecule_id


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + 100*sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, molecule_partitions, num_molecules ,placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['molecule_partitions']:molecule_partitions})
    feed_dict.update({placeholders['num_molecules']:num_molecules})

    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def tensor_diff(self, input_tensor):
    #builds a matrix like (for an example for 3 molecules):
    #  [1   0  0]
    #  [-1  1  0]
    #  [0  -1  1]
    # and multiplies it by the input vector to get a difference vector
        
    A = tf.eye(self.num_molecules)
    B = tf.pad(tf.negative(tf.eye(tf.subtract(self.num_molecules,tf.constant(1)))), tf.constant([[1, 0,], [0, 1]]), "CONSTANT")
    d_tensor = tf.add(A,B)
    out = tf.matmul(d_tensor,input_tensor, name="output_after_tensorDiff")
    return out