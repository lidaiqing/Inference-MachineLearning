import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def getRandomBatch(trainData, trainTarget, size):
    idx = np.random.choice(trainData.shape[0], size, replace=False)
    return trainData[idx,:], trainTarget[idx,:]

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def layerBlock(x, n_units):
    '''
    x:input tensor
    n_units:number of hidden units in this layer
    '''
    n_input = x.get_shape().as_list()[1]
    XavierSTD = np.sqrt(3.0 / (n_input + n_units))
    W = tf.Variable(tf.truncated_normal(shape=[n_input, n_units], stddev=XavierSTD), name="weights")
    b = tf.Variable(tf.zeros([n_units]), name="biases")
    z = tf.add(tf.matmul(x, W), b)
    return z

def layerBlock_decay(x, n_units, decay_rate):
    '''
    x:input tensor
    n_units:number of hidden units in this layer
    '''
    n_input = x.get_shape().as_list()[1]
    XavierSTD = np.sqrt(3.0 / (n_input + n_units))
    W = tf.Variable(tf.truncated_normal(shape=[n_input, n_units], stddev=XavierSTD), name="weights")
    b = tf.Variable(tf.zeros([n_units]), name="biases")
    weight_decay = tf.mul(tf.nn.l2_loss(W), decay_rate)
    tf.add_to_collection('losses',weight_decay)
    z = tf.add(tf.matmul(x, W), b)
    return z, W
