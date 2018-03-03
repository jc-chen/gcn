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

def add_sample(url,inputMatrix,outputArray,i,n,d,A): 
    
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
        temp = np.array([0 for i in range(d)]);
        temp[0:len(vertices)] = vertices;
        inputMatrix[i*d:(i+1)*d][0:len(vertices)] = temp #[vertices[v] for v in vertices,0 for j in range(len(inputMatrix[i])-len(vertices)-1)];
        outputArray[i*d:(i+1)*d] = [float(properties[15]) for j in range(d)]

        tempA = np.zeros((d,d)); #Adjacency matrix

        #populate the adjacency matrix
        for tupl in edges:
            tuple_list = list(tupl);
            v_i = tuple_list[0];
            v_j = tuple_list[1];
            tempA[v_i][v_j] = 1;
            tempA[v_j][v_i] = 1;
        A[i*d:(i+1)*d,i*d:(i+1)*d] = tempA;
        return;

def load_data2():
    """Load data."""
########### whta I should do instead:
# using some array with append()
#iterate over each molecule
#for each molecule, obtain # of atoms (m) from first line
# append the features (e.g. atomic number) into X so each ROW of x is an atom and column are features like number e-, atomic number
# X ends up being a ton of atoms bc all the molecules get appended after each other
# make and save the adjacency matrix (mxm)
# for the y's...remember to repeat each y-value m times
#separate into training and validation sets 

#then create some separate arrays in same fashion for the test data
#at the end create the gigantic adjacency matrix for everything
#mask: leave as is


    d = 26 #no. features (size of largest molecule)
    n = 1000 #no samples
    X = np.zeros((n*d,d));
    Y = np.zeros(n*d);
    A = np.zeros((n*d,n*d));
    # let v be the number at which we start using validation
    # let r be the number at which we start test data
    v = 7*n/10;
    r = 9*n/10;
    path = "../tem/";

    i=0;
    for file in os.listdir(path):
        #print(file)
        add_sample(path+file,X,Y,i,n,d,A);
        i += 1;

    adj = sp.csr_matrix(A);
    x = X[0:d*r]
    tx = X[d*r:]
    y = Y[0:d*r]
    ty = Y[d*r:]
    x = sp.coo_matrix(x)
    tx = sp.coo_matrix(tx)
    #y=sp.coo_matrix(y)
    #ty=sp.coo_matrix(ty)
    allx = x;
    ally = y;

    features = sp.vstack((allx, tx)).tolil()
    labels = Y #np.vstack((ally, ty))

    idx_test = range(len(y),len(ally))
    idx_train = range(len(y))
    idx_val = range(len(y),r)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
   
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


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
    r_inv = np.power(rowsum, -1).flatten()
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
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
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
