from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils2 import *
from gcn.models import JCNN

from train import evaluate



sess = tf.Session()
saver = tf.train.import_meta_graph('./trained-model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))



test_cost, test_acc, test_mae, test_duration = evaluate(features, support, y_test, molecule_partitions, num_molecules, placeholders,mask=test_mask)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy= ", str(test_acc), "mae= ", str(test_mae), "time=", "{:.5f}".format(test_duration))





exit()
