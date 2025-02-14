import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, os, Cython
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, rdmolops
import molmod as mm;
from molmod.periodic import periodic
import tensorflow as tf
from random import shuffle

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

def pickle_file(path, *args):
    for arg in args:
        file_name = './loaded_data/' + path + str(arg[0]) + '.txt'
        with open(file_name,'wb') as file:
            pkl.dump(arg[1],file,protocol=2)
    return 

def load_pickled(path, *args):
    a=[]

    for arg in args:
        print("Loading "+str(arg)+".txt now")
        file_name = './loaded_data/' + path + str(arg) + '.txt'
        with open(file_name,'rb') as file:
            a += [pkl.load(file)]
    return a

#DEPRECATED
# def add_sample_molmod(url,features,target,A,sizes,num_molecules,elements_info):
#     #extract information from xyz file
#     try:
#         mol = mm.Molecule.from_file(url);
#         properties = [];

#         with open(url,'r') as file:
#             for row in file:
#                 properties += row.split();

#         #mol.write_to_file("new.xyz");
#         mol.graph = mm.MolecularGraph.from_geometry(mol);
#         vertices = mol.graph.numbers;
#         edges = mol.graph.edges;
#         d = len(vertices) #the size of each molecule
#         partial_charges=properties[22:(23+5*(d-1)):5]
#         atomic_number_mean = elements_info[1]
#         atomic_number_stdev = elements_info[2]

#         # Find Benzene
#         # if (d==12):
#         #     if (list(vertices).count(1) == 6):
#         #         if (list(vertices).count(6) == 6):
#         #             print(url)
#         #             return features, target, A, sizes, num_molecules 
#         # else:
#         #     return features, target, A, sizes, num_molecules


#         # Targets
#         dipole_moment = float(properties[6])
#         polarizability = float(properties[7])
#         homo = float(properties[8])
#         lumo = float(properties[9])
#         gap = float(properties[10])
#         r2 = float(properties[11])
#         zpve = float(properties[12])
#         U0 = float(properties[13])
#         internal_energy = float(properties[14])
#         enthalpy = float(properties[15])
#         free_nrg = float(properties[16])
#         heat_capacity = float(properties[17])
  
#         target.append([heat_capacity])
#         #target.append([dipole_moment,polarizability,homo,lumo,gap,
#         #    r2,zpve,U0,internal_energy,enthalpy,free_nrg,heat_capacity])


#         tempA = np.zeros((d,d)); #Adjacency matrix

#         #populate the adjacency matrix with intermolecular distances in terms of 1/r^2
#         for tupl in edges:
#             tuple_list = list(tupl);
#             v_i = tuple_list[0];
#             v_j = tuple_list[1];
#             tempA[v_i][v_j] = 1.0/pow(mol.distance_matrix[v_i][v_j],2)
#             tempA[v_j][v_i] = 1.0/pow(mol.distance_matrix[v_i][v_j],2)

#         A.append(sp.coo_matrix(tempA))
        
#         # Write features
#         f = 11
#         tempfeatures = [[0]*f for _ in range(d)]; # d=#nodes,  f=#features available
#         #Structure of the features matrix: (in a row)
#         # atomic_no, H, C, N, O, F, #H, vdw radius, partial charge, acceptor, donor
#         # int, one-hot (5 cols), int, float, float, one-hot (2 cols)
#         for atom in range(len(vertices)):
#             tempfeatures[atom][0] = (float(vertices[atom])-atomic_number_mean)/atomic_number_stdev
#             tempfeatures[atom][1] = int(vertices[atom]==1) #H
#             tempfeatures[atom][2] = int(vertices[atom]==6) #C
#             tempfeatures[atom][3] = int(vertices[atom]==7) #N
#             tempfeatures[atom][4] = int(vertices[atom]==8) #O
#             tempfeatures[atom][5] = int(vertices[atom]==9) #F
#             tempfeatures[atom][6] = 1.0/(list(vertices).count(1)+1.0) #number of H
#             tempfeatures[atom][7] = float(periodic[vertices[atom]].vdw_radius)
#             tempfeatures[atom][8] = float(partial_charges[atom]) #Mulliken partial charge
#             tempfeatures[atom][9] = int(float(partial_charges[atom]) >0.0)
#             tempfeatures[atom][10] = int(float(partial_charges[atom]) <0.0)
#         if (num_molecules == 0):
#             sizes = sizes + [d-1]
#         else:
#             sizes = sizes + [d]
#         num_molecules=num_molecules+1
#         if (num_molecules % 5000 == 0):
#             print("On the "+str(num_molecules)+"th molecule")

        
#         features+=tempfeatures
#         return features, target, A, sizes, num_molecules
#     except Exception as e:
#         #Write exception to file
#         print(str(e))
#         with open("analysis/problem_files.txt","a+") as file:
#             file.write(str(num_molecules)+" :  "+str(url)+ "   " + str(e) + "\n")
#         return features, target, A, sizes, num_molecules

def add_sample(url,features,target,As,sizes,num_molecules,elements_info):
    #extract information from xyz file
    #try:
    if (True):
        properties = [];
        with open(url,'r') as file:
            for row in file:
                properties += row.split()
        
        SMILES = properties[-4]
        INCHI = properties[-2]
        m = Chem.MolFromSmiles(SMILES)
        m = Chem.AddHs(m)
        
        vertices = m.GetAtoms()
        #edges = m.GetBonds()
        d = len(vertices)
        partial_charges=properties[22:(23+5*(d-1)):5]
        atomic_number_mean = elements_info[1]
        atomic_number_stdev = elements_info[2]

        #pc = m.ComputeGasteigerCharges()
        #print(pc,partial_charges)

        # Targets
        dipole_moment = float(properties[6])
        polarizability = float(properties[7])
        homo = float(properties[8])
        lumo = float(properties[9])
        gap = float(properties[10])
        r2 = float(properties[11])
        zpve = float(properties[12])
        U0 = float(properties[13])
        internal_energy = float(properties[14])
        enthalpy = float(properties[15])
        free_nrg = float(properties[16])
        heat_capacity = float(properties[17])
  
        target.append([heat_capacity])
        #target.append([dipole_moment,polarizability,homo,lumo,gap,
        #    r2,zpve,U0,internal_energy,enthalpy,free_nrg,heat_capacity])

        #populate the adjacency matrix and write features
        tempA = rdmolops.GetDistanceMatrix(m) #np.zeros((d,d)); #Adjacency matrix
        tempBO = np.zeros((d,d))
        tempAR = np.zeros((d,d))
        f = 8
        tempfeatures = [[0]*f for _ in range(d)]; # d=#nodes,  f=#features available

        dic = {'C':6, 'H':1, 'O':8, 'N':7, 'F':9}
        for atom in vertices:
            # Get features of the atom
            v_i = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            valence = atom.GetTotalValence()
            hybrid = int(atom.GetHybridization())
            atom_aromatic = atom.GetIsAromatic()
            ring = atom.IsInRing()
            rad = atom.GetNumRadicalElectrons()

            # Get bonds
            linked_atoms =[x.GetIdx() for x in atom.GetNeighbors()]

            # Populate adjacency matrix
            tempBO[v_i][v_i] = 10 #arbitrarily large number to indicate connectivity to self
            for v_j in linked_atoms:
                bond_order = m.GetBondBetweenAtoms(v_i,v_j).GetBondTypeAsDouble()
                #bond_length = m.GetBondBetweenAtoms(v_i,v_j).GetBondLength()
                bond_aromatic = m.GetBondBetweenAtoms(v_i,v_j).GetIsAromatic()
                tempBO[v_i][v_j] = bond_order
                tempAR[v_i][v_j] = bond_aromatic
            
            # Write features
            tempfeatures[v_i][0] = (atomic_num - atomic_number_mean)/atomic_number_stdev
            tempfeatures[v_i][1] = int(atomic_num==1) #H
            tempfeatures[v_i][2] = int(atomic_num==6) #C
            tempfeatures[v_i][3] = int(atomic_num==7) #N
            tempfeatures[v_i][4] = int(atomic_num==8) #O
            tempfeatures[v_i][5] = int(atomic_num==9) #F
            tempfeatures[v_i][6] = int(atom_aromatic)
            tempfeatures[v_i][7] = hybrid
            #tempfeatures[v_i][8] = ring #float(partial_charges[atom]) #Mulliken partial charge
            #tempfeatures[v_i][9] = degree
            #tempfeatures[v_i][10] = valence

        As[0].append(sp.coo_matrix(tempA))
        As[1].append(sp.coo_matrix(tempBO))
        As[2].append(sp.coo_matrix(tempAR))
        
        if (num_molecules == 0):
            sizes = sizes + [d-1]
        else:
            sizes = sizes + [d]
        num_molecules=num_molecules+1

        if (num_molecules % 5000 == 0):
            print("On the "+str(num_molecules)+"th molecule")
        
        features+=tempfeatures
        return features, target, As, sizes, num_molecules
    # except Exception as e:
    #     #Write exception to file
    #     print(str(e))
    #     with open("analysis/problem_files.txt","a+") as file:
    #         file.write(str(num_molecules)+" :  "+str(url)+ "   " + str(e) + "\n")
    #     return features, target, A, sizes, num_molecules



def load_data3(data_path, pklpath, pklflag=0, loadflag=0):
    """Load data."""

    if (loadflag==1):
        return load_pickled(pklpath,'adj','features','y_train',
            'y_val', 'y_test','train_mask','val_mask','test_mask','molecule_partitions','num_molecules')

    features = [] #features of each node
    A=[] #bond "graph distance" for now  #list of graph adjacency matrices; each entry is the adjacency matrix for one molecule
    BO = [] #adjacency matrix of bond order
    AR = [] #adjacency matrix - isaromatic

    sizes = [] #list of sizes of molecules; each entry is the size of a molecule
    num_molecules = 0
    target = [] #list of "y's" - each entry is an "answer" for a molecule


    # Info for standardizing data
    elements = np.array([1,6,7,8,9])
    elements_mean = np.mean(elements)
    elements_stdev = np.std(elements)
    elements_all = [elements, elements_mean, elements_stdev]

    for file in os.listdir(data_path):
        features, target, [A,BO,AR], sizes, num_molecules = add_sample(data_path+file,features,target,[A,BO,AR],sizes,num_molecules,elements_all)

    print("Total molecules",num_molecules)

    molecule_partitions=np.cumsum(sizes) #to get partition positions


    # Divide into train, validation, test sets
    randomized_order = list(range(num_molecules))
    shuffle(randomized_order)
    train_size = int(num_molecules*3/5)
    val_size = train_size + int(num_molecules/5)
    idx_train = randomized_order[0:train_size]
    idx_val = randomized_order[train_size:val_size]
    idx_test = randomized_order[val_size:]

    print([val_size,train_size])
    tar = np.array(target)
    target_mean = np.mean(target,axis=0)
    target_std = np.std(target,axis=0)

    adj = [sp.csr_matrix(sp.block_diag(BO))]#,sp.csr_matrix(sp.block_diag(AR))]
    labels = np.array(target)

    # histy,histx = np.histogram(labels,bins=20)
    # fig,ax = plt.subplots()
    # ax.plot(histx[:-1],histy)
    # plt.show()

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

    print("About to write to file")

    if (pklflag):
        pickle_file(pklpath,('adj', adj), ('features', feats), ('y_train', y_train), ('y_val', y_val), ('y_test', y_test), 
            ('train_mask', train_mask), ('val_mask', val_mask), ('test_mask', test_mask),
            ('molecule_partitions',molecule_partitions),('num_molecules',num_molecules))

        print("finished writing to file")

    #print([adj, feats, y_train, y_val, y_test, train_mask, val_mask, test_mask, molecule_partitions, num_molecules])

    return [adj, feats, y_train, y_val, y_test, train_mask, val_mask, test_mask, molecule_partitions, num_molecules]

def load_data_new(path,flag=0):
    """Load data."""

    features = [] #features of each node
    A=[] #list of graph adjacency matrices; each entry is the adjacency matrix for one molecule
    sizes = [] #list of sizes of molecules; each entry is the size of a molecule
    num_molecules = 0
    target = [] #list of "y's" - each entry is an "answer" for a molecule


    # Info for standardizing data
    elements = np.array([1,6,7,8,9])
    elements_mean = np.mean(elements)
    elements_stdev = np.std(elements)
    elements_all = [elements, elements_mean, elements_stdev]

    for file in os.listdir(path):
        print(str(file))
        features, target, A, sizes, num_molecules = add_sample(path+file,features,target,A,sizes,num_molecules,elements_all)

    print("Total molecules",num_molecules)

    molecule_partitions=np.cumsum(sizes) #to get partition positions
    #n = molecule_partitions[-1]+1 #total sum of all nodes

    adj = sp.csr_matrix(sp.block_diag(A))

    y = np.array(target)

    print("defined labels")

    feats = sp.coo_matrix(np.array(features)).tolil()

    #TODO, decide how to handle this
    # print("About to write to file")
    # write_file(('target_mean', target_mean), ('target_stdev', target_stdev), ('adj', adj),
    #     ('features', feats), ('y_train', y_train), ('y_val', y_val), ('y_test', y_test), 
    #     ('train_mask', train_mask), ('val_mask', val_mask), ('test_mask', test_mask),
    #     ('molecule_partitions',molecule_partitions),('num_molecules',num_molecules))

    print("Finished writing to file")

    return [adj, feats, y, molecule_partitions, num_molecules]


def load_data_test(data_path, flag=0):
    """Load data."""
    features = [] #features of each node
    A=[] #list of graph adjacency matrices; each entry is the adjacency matrix for one molecule
    sizes = [] #list of sizes of molecules; each entry is the size of a molecule
    num_molecules = 0
    target = [] #list of "y's" - each entry is an "answer" for a molecule


    # Info for standardizing data
    elements = np.array([1,6,7,8,9])
    elements_mean = np.mean(elements)
    elements_stdev = np.std(elements)
    elements_all = [elements, elements_mean, elements_stdev]

    for file in os.listdir(data_path):
        features, target, A, sizes, num_molecules = add_sample(data_path+file,features,target,A,sizes,num_molecules,elements_all)

    print("Total molecules",num_molecules)

    molecule_partitions=np.cumsum(sizes) #to get partition positions

    adj = sp.csr_matrix(sp.block_diag(A))
    labels = np.array(target)

    feats = sp.coo_matrix(np.array(features)).tolil()

    return [adj, feats, labels, molecule_partitions, num_molecules]


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
    """Preprocessing of a list of adjacency matrices for simple GCN model and conversion to tuple representation."""
    for i in range(len(adj)):
        adj_normalized = normalize_adj(adj[i])#+ 10.0*sp.eye(adj[i].shape[0]))
        adj[i] = sparse_to_tuple(adj_normalized)
    return adj


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
    #builds an m by m matrix like (for example for 3 molecules):
    #  [1   0  0]
    #  [-1  1  0]
    #  [0  -1  1]
    # and multiplies it by the input vector to get a difference vector
        
    A = tf.eye(self.num_molecules)
    B = tf.pad(tf.negative(tf.eye(tf.subtract(self.num_molecules,tf.constant(1)))), tf.constant([[1, 0,], [0, 1]]), "CONSTANT")
    d_tensor = tf.add(A,B)
    out = tf.matmul(d_tensor,input_tensor, name="output_after_tensorDiff")
    return out


def sparse_matrix_to_sparse_tensor(coo):
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def print_learn_rate(rate):
    print("......\n......\n ......Changing learning rate to: ", rate, "\n......\n......")
    return