from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils2 import *
from gcn.models import GCN, MLP, JCNN


def test(n, input_tensor):
    #builds an m by m matrix like (for example for 3 molecules):
    #  [1   0  0]
    #  [-1  1  0]
    #  [0  -1  1]
    # and multiplies it by the input vector to get a difference vector
        
    A = tf.eye(n)
    B = tf.pad(tf.negative(tf.eye(tf.subtract(n,tf.constant(1)))), tf.constant([[1, 0,], [0, 1]]), "CONSTANT")
    d_tensor = tf.add(A,B)
    out = tf.matmul(d_tensor,input_tensor, name="output_after_tensorDiff")
    return out

def turmoil_func():
    n=10
    m=4
    p=3
    partits = tf.constant([2,3,7,9])
    outputter = tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[0,0,0],[0,1,2],[3,4,5],[6,7,8],[0,1,2],[0.5,0,1]])
    outputter = tf.cumsum(outputter)
    outputter = tf.gather(outputter,partits)
    outputter = test(m,outputter)
    return outputter

def wtf():
    #run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outs = sess.run([turmoil_func()],feed_dict=None)
        print(outs)
    exit()

def squared_error(preds, labels, mask):
    """L2 loss refactored to incorporate masks"""
    mask = tf.transpose(mask)
    mask = tf.cast(mask,dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,2])
    mask /= tf.reduce_mean(mask)
    loss = tf.losses.mean_squared_error(labels,preds,reduction=tf.losses.Reduction.NONE)
    loss *= mask
    return tf.reduce_mean(loss)

def wtf2():
    #run the graph
    pred = tf.constant([[5.],[9],[3]])
    labs = tf.constant([[2.],[7],[10]])
    mask = tf.constant([1,1,0])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outs = sess.run([squared_error(pred,labs,mask)],feed_dict=None)
        print(outs)
    exit()



# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,molecule_partitions, num_molecules = load_data3()

print("mewoers")
print("finished loading data")
print(y_train[0:10])

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'jcnn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 26, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 18, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 38, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 36, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 30, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('hidden6', 30, 'Number of units in hidden layer 6.')
flags.DEFINE_integer('hidden7', 28, 'Number of units in hidden layer 7.')
flags.DEFINE_integer('hidden8', 28, 'Number of units in hidden layer 8.')
flags.DEFINE_integer('hidden9', 30, 'Number of units in hidden layer 9.')
flags.DEFINE_integer('hidden10', 24, 'Number of units in hidden layer 10.')
flags.DEFINE_integer('hidden11', 24, 'Number of units in hidden layer 11.')
flags.DEFINE_integer('hidden12', 24, 'Number of units in hidden layer 12.')
flags.DEFINE_integer('node_output_size', 10, 'Number of hidden features each node has prior to readout')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
elif FLAGS.model == 'jcnn':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = JCNN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None,y_train.shape[1]), name='the_labels_meow'),
    'labels_mask': tf.placeholder(tf.int32, name='the_mask_of_labels'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout_meow'),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'molecule_partitions': tf.placeholder(tf.int32,shape=(molecule_partitions.shape)),
    'num_molecules': tf.placeholder(tf.int32,shape=())
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
#input_dim is like...if you have k features for each node, then input_dim=k


# Initialize session
sess = tf.Session()

# Tensorboard stuff
summary_ops = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('../tensorboard/',sess.graph)


# Define model evaluation function
def evaluate(features, support, labels, mask, molecule_partitions, num_molecules, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, molecule_partitions, num_molecules, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

#summary_writer = tf.train.SummaryWriter('/tmp/logs', sess.graph_def)


# Train model
tf.summary.scalar('second', tf.Variable(5))

for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, molecule_partitions, num_molecules, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,summary_ops,model.outputs], feed_dict=feed_dict)

    summary_writer.add_summary(outs[3], epoch)
    #summary_writer.flush()

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, molecule_partitions, num_molecules, placeholders)
    cost_val.append(cost)

    #print(y_train[1])

    #print(type(outs[4][1]),outs[4])
    # Log a summary ever 10 steps
    #if epoch % 10 == 0:
    #    summary_writer.add_summary(some_kind_of_summary, epoch)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    print(outs[4][0],y_train[0])
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, molecule_partitions, num_molecules, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
