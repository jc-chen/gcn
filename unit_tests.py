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
    outputter = tf.constant([[-1.0,2,-3],[4,-5,6],[-7,8,9],[-10,-11,-12],[0,0,0],[0,1,2],[3,4,5],[6,7,8],[0,1,2],[0.5,0,1]])
    outputter = tf.cumsum(outputter)
    outputter = tf.gather(outputter,partits)
    outputter = test(m,outputter)
    return outputter

def testf():
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
    loss = tf.multiply(loss,mask)
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    mask=tf.transpose(mask)
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,labels.shape[1]])
    mask /= tf.reduce_mean(mask)

    #mnabserr = tf.metrics.mean_absolute_error(labels,preds)
    #accuracy_all = tf.multiply(mnabserr,mask)
    #accuracy_all *= mask
    
    loss = tf.losses.mean_squared_error(labels,preds,reduction=tf.losses.Reduction.NONE)
    loss = tf.multiply(loss,mask)
    return tf.reduce_mean(loss)

def tf2():
    #run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pred = tf.constant([[5.],[9],[3]])
        labs = tf.constant([[3.],[10],[10]])
        mask = tf.constant([1,0,1])
        outs = sess.run([masked_accuracy(pred,labs,mask)],feed_dict=None)
        print(outs)
    exit()