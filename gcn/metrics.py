import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
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

def square_error(preds, labels, mask):
    """L2 loss refactored to incorporate masks"""
    #n = len(preds)
    mask = tf.cast(mask,dtype=tf.float32)
    mask = tf.expand_dims(mask,-1)
    mask = tf.tile(mask,[1,labels.shape[1]])
    mask /= tf.reduce_mean(mask)
    loss = tf.losses.mean_squared_error(labels,preds,reduction=tf.losses.Reduction.NONE)
    loss = tf.multiply(loss,mask)
    return tf.reduce_mean(loss)
