from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils2 import *
from gcn.models import JCNN


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
load_previous = 1
#########[target_mean,target_stdev,adj,features,y_train,y_val,y_test,train_mask,val_mask,test_mask,molecule_partitions,num_molecules]=load_data3(data_path,load_previous)
data_path = '../tem_1000/' #'../batches/batch0/'

[adj,features,y_train,y_val,y_test,train_mask,val_mask,test_mask,molecule_partitions,num_molecules]=load_data3(data_path,load_previous)
#[adj_new,features_new,y_new,molecule_partitions_new,num_molecules_new]=load_data_new(data_path_new)

#support_new = [preprocess_adj(adj_new)]
#features_new = preprocess_features(features_new)

print("Finished loading data!")

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

# Some preprocessing
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
    'labels': tf.placeholder(tf.float32, shape=(None,y_train.shape[1]), name='the_labels_meow'),
    'labels_mask': tf.placeholder(tf.int32, name='the_mask_of_labels'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout_meow'),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'molecule_partitions': tf.placeholder(tf.int32),
    'num_molecules': tf.placeholder(tf.int32,shape=())
}





# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
#input_dim is like...if you have k features for each node, then input_dim=k


# Define model evaluation function
def evaluate(features, support, labels, molecule_partitions, num_molecules, placeholders, mask=None):

    if mask is None:
        mask = np.array(np.ones(labels.shape[0]), dtype=np.bool)

    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, molecule_partitions, num_molecules, placeholders)
    outs_val = sess.run([model.loss, model.accuracy,model.mae], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)




sess = tf.Session()

sess.run(tf.global_variables_initializer())

nsaver = tf.train.import_meta_graph('./trained-model.meta')
nsaver.restore(sess,tf.train.latest_checkpoint('./'))


test_cost, test_acc, test_mae, test_duration = evaluate(features, support, y_test, molecule_partitions, num_molecules, placeholders,mask=test_mask)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy= ", str(test_acc), "mae= ", str(test_mae), "time=", "{:.5f}".format(test_duration))




exit()




# Initialize session
saver = tf.train.Saver()
sess = tf.Session()

# Tensorboard stuff
summary_ops = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('../tensorboard/',sess.graph)


# Init variables
sess.run(tf.global_variables_initializer())

#normalize targets in model
[m,s]=sess.run([model.get_mean,model.get_std], feed_dict={placeholders['labels']: y_train, placeholders['labels_mask']: train_mask})

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
    outs = sess.run([model.opt_op, model.loss, model.accuracy,summary_ops,model.outputs,model.mae], feed_dict=feed_dict)

    summary_writer.add_summary(outs[3], epoch)
    #summary_writer.flush()

    # Validation
    cost, acc, mae, duration = evaluate(features, support, y_val, molecule_partitions, num_molecules, placeholders, mask=val_mask)
    cost_val.append(cost)

    if (epoch == 50):
        FLAGS.learning_rate = 2.0
        print("Changing learning rate to: ", FLAGS.learning_rate)
    if (epoch == 120):
        FLAGS.learning_rate = 1.0
    if (epoch == 300):
        FLAGS.learning_rate = 0.5
        print("Changing learning rate to: ", FLAGS.learning_rate)
    if (epoch == 1200):
        FLAGS.learning_rate = 0.1
        print("Changing learning rate to: ", FLAGS.learning_rate)
    if (epoch == 1200):
        FLAGS.learning_rate = 0.05
        print("Changing learning rate to: ", FLAGS.learning_rate)
    if (epoch == 1500):
        FLAGS.learning_rate = 0.01
        print("Changing learning rate to: ", FLAGS.learning_rate)
    if (epoch == 3000):
        FLAGS.learning_rate = 0.001
        print("Changing learning rate to: ", FLAGS.learning_rate)

    # Log a summary ever 10 steps
    #if epoch % 10 == 0:
    #    summary_writer.add_summary(some_kind_of_summary, epoch)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "val_loss=", "{:.5f}".format(cost), "train_acc= ", str(outs[2]),        
          " val_acc= ", str(acc),"train_mae= ", str(outs[5]), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break


print("Optimization Finished!")


plswork = sess.run(model.vars,feed_dict=feed_dict)


saver.save(sess,'./trained-model')


exit()

# Testing
test_cost, test_acc, test_mae, test_duration = evaluate(features, support, y_test, molecule_partitions, num_molecules, placeholders,mask=test_mask)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy= ", str(test_acc), "mae= ", str(test_mae), "time=", "{:.5f}".format(test_duration))

#New testing 
# test_cost, test_acc, test_mae, test_duration = evaluate(features_new, support_new, y_new, molecule_partitions_new, num_molecules_new, placeholders)
# print("New test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy= ", str(test_acc), "mae= ", str(test_mae), "time=", "{:.5f}".format(test_duration))


Costs_file = open("analysis/costs.txt","a+")
Costs_file.write("mu\n")
Costs_file.write("Epochs: " + str(epoch + 1) + "\ntrain_loss= " + str(outs[1]) + 
    "      val_loss= " + str(cost) + "\ntrain_acc= " + str(outs[2]) + "\nval_acc= " + 
    str(acc) + "\ntrain_mae= " + str(outs[5]))
Costs_file.write("\n")
Costs_file.write("Test cost= " + str(test_cost) + "\nTest_acc= " +str(test_acc))
Costs_file.write("\nTest mae= " + str(test_mae))
Costs_file.write("\n\n")
Costs_file.close()