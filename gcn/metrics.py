import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy_new(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.logical_and(tf.less(preds,labels*1.1),tf.greater(preds,labels*0.9))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.logical_and(tf.less(preds,labels*1.1),tf.greater(preds,labels*0.9))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    print(labels.shape)
    print("MROW")
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,8])
    mask /= tf.reduce_mean(mask)
    print(mask.shape)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def square_error(preds, labels, mask):
    """L2 loss refactored to incorporate masks"""
    #n = len(preds)
    print("MEow Mwoe")
    mask = tf.cast(mask,dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,8])
    mask /= tf.reduce_mean(mask)
    loss = tf.losses.mean_squared_error(labels,preds,reduction=tf.losses.Reduction.NONE)
    loss *= mask
    return tf.reduce_mean(loss)