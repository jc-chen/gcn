from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils2 import *
from gcn.models import JCNN

import argparse
import os

print("this is loading ")

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'jcnn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 3.0, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 5000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 36, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 30, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 34, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 28, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 22, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('hidden6', 30, 'Number of units in hidden layer 6.')
flags.DEFINE_integer('hidden7', 28, 'Number of units in hidden layer 7.')
flags.DEFINE_integer('hidden8', 28, 'Number of units in hidden layer 8.')
flags.DEFINE_integer('hidden9', 30, 'Number of units in hidden layer 9.')
flags.DEFINE_integer('hidden10', 24, 'Number of units in hidden layer 10.')
flags.DEFINE_integer('hidden11', 24, 'Number of units in hidden layer 11.')
flags.DEFINE_integer('hidden12', 24, 'Number of units in hidden layer 12.')
flags.DEFINE_integer('node_output_size', 10, 'Number of hidden features each node has prior to readout')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


load_previous = 0
pklpath ='7000/b1/'
data_path = '../benzene/' #'../' + pklpath
model_path = './models/' + pklpath + 'name'
[adj,features,y_test,molecule_partitions,num_molecules]=load_data_new(data_path,load_previous)

print("Finished loading data")

features = preprocess_features(features)
if FLAGS.model == 'jcnn':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = JCNN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None,y_test.shape[1]), name='the_labels_meow'),
    'labels_mask': tf.placeholder(tf.int32, name='the_mask_of_labels'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout_meow'),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'molecule_partitions': tf.placeholder(tf.int32),
    'num_molecules': tf.placeholder(tf.int32,shape=())
}



# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
#input_dim is like...if you have k features for each node, then input_dim=k

sess = tf.Session()

sess.run(tf.global_variables_initializer())

nsaver = tf.train.Saver()
nsaver.restore(sess,model_path)


# Define model evaluation function
def evaluate(features, support, labels, molecule_partitions, num_molecules, placeholders, mask=None):

    if mask is None:
        mask = np.array(np.ones(labels.shape[0]), dtype=np.bool)

    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, molecule_partitions, num_molecules, placeholders)
    outs_val = sess.run([model.loss, model.accuracy,model.mae,model.predict()], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)


test_cost, test_acc, test_mae, test_prediction, test_duration = evaluate(features, support, y_test, molecule_partitions, num_molecules, placeholders)
print("Test set results:", "prediction= ", str(test_prediction),
	"  cost=", "{:.5f}".format(test_cost), "accuracy= ", str(test_acc), "  mae= ", 
	str(test_mae), "  time=", "{:.5f}".format(test_duration))


