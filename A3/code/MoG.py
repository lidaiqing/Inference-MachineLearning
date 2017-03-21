import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *


data = np.load("../data/data2D.npy")
DATA_DIMENSION = 2
points_2d = data
x = points_2d[:,0]
y = points_2d[:,1]
color_list = ["g", "b", "r", "y", "k"]

def get_covariance_mat(sigma):

    return tf.matrix_diag(tf.square(sigma))

# oneDivSqrtTwoPI = 1 / math.sqrt(2*tf.math.pi)
def get_mul_normal(x, mu, sigma):
    '''
    x should be in [N, DATA_DIMENSION]. mu should be in [K, DATA_DIMENSION].
    sigma should be in [K, DATA_DIMENSION]

    It returns a [N, K] matrix representing the pdf value
    '''

    print x.get_shape().as_list()
    print mu.get_shape().as_list()
    print sigma.get_shape().as_list()

    assert x.get_shape().as_list()[1] == mu.get_shape().as_list()[1]
    assert mu.get_shape().as_list()[1] == DATA_DIMENSION

    N = x.get_shape().as_list()[0]
    K = mu.get_shape().as_list()[0]
    D = x.get_shape().as_list()[1]

    # print type([N, K, D][0])
    # print type([1,2,3][0])
    # Covariance shouls be in [K, DATA_DIMENSION, DATA_DIMENSION]
    covariance = get_covariance_mat(sigma)

    tmp = tf.tile(x, [1, K])
    x_sub_mu = tf.reshape(tmp, [-1, K, D]) - mu

    # print x_sub_mu.get_shape().as_list()

    expo = tf.exp(-0.5 * tf.reduce_sum(tf.multiply(tf.transpose(tf.matmul(tf.transpose(x_sub_mu,[1,0,2]),tf.matrix_inverse(covariance)),[1,0,2]),x_sub_mu),2))
    ret = tf.pow(2 * np.pi, tf.cast(-tf.shape(x)[1], tf.float32) / 2) * tf.pow(tf.sqrt(tf.reduce_sum(tf.square(covariance))), - 0.5) * expo

    return ret

def buildGraph_mog_Adam(K, learning_rate):
    # Variable creation
    points = tf.placeholder(tf.float32, [None, 2], name='input_points')
    mu = tf.Variable(tf.truncated_normal(shape=[K,2], stddev=1), name='mu')
    # sigma = tf.Variable(tf.truncated_normal(shape=[K,2], stddev=0.5), name='sigma')
    sigma = tf.Variable(tf.ones([K,2]), name='sigma')
    pi =  tf.Variable(tf.div(tf.ones([K]),K), name="pi")

    N = tf.shape(points)[0]

    pdf = get_mul_normal(points, mu, sigma)
    print tf.shape(pdf)
    loss = -tf.reduce_sum(reduce_logsumexp(pi * pdf))

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(loss=loss)
    return points, mu, loss, train, pdf

learning_rate = 0.1
K = 3
points, mu, loss, train, pdf = buildGraph_mog_Adam(K, learning_rate)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
loss_recorder = np.array([])

numIteration = 500
for itr in range(numIteration):
    loss_, _, prob, centroid = sess.run([loss, train, pdf, mu], feed_dict={points: points_2d})
    loss_recorder = np.append(loss_recorder, loss_)
    if itr % 100 == 0:
        print("Iteration#: %d, loss: %0.2f"%(itr, loss_))
# plt.plot(np.arange(numIteration), loss_recorder, 'g')
# #plt.axis([0,500, 0, 2])
# plt.title("Total loss VS number of updates for learning_rate = %0.3f"%(learning_rate))
# plt.show()
# print prob[0:10,:]
        assign = np.argmax(prob, axis = 1)
        colors = [color_list[assign[i]] for i in range(len(x))]
        plt.scatter(x,y,c=colors)
# print centroid.shape
        colors = ['k' for i in range(centroid.shape[0])]
        plt.scatter(centroid[:,0],centroid[:,1],c=colors)
        #plt.axis([0,500, 0, 2])
        plt.title("K mean plots for k = %d"%(K))
        plt.show()
