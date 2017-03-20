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

def logP(x, mu, stddev):
    x_sub_mu = tf.reduce_sum((tf.expand_dims(mu,2) - tf.expand_dims(tf.transpose(x), 0)) ** 2, 1)
    variance = tf.square(stddev)
    ret = -0.5 * tf.cast(tf.rank(x), tf.float32) * tf.log(2 * np.pi * variance) - 0.5 * (tf.multiply(1/variance,x_sub_mu))

    return ret

def buildGraph_mog_Adam(K, learning_rate):
    # Variable creation
    points = tf.placeholder(tf.float32, [None, 2], name='input_points')
    mu = tf.Variable(tf.truncated_normal(shape=[K,2], stddev=1), name='mu')
    # sigma = tf.Variable(tf.truncated_normal(shape=[K,2], stddev=0.5), name='sigma')
    stddev = tf.Variable(tf.exp(tf.truncated_normal(shape=[K,1], stddev=0.5)), name='sigma')
    # pi =  tf.Variable(tf.div(tf.ones([K]),K), name="pi")
    phi = tf.Variable(tf.div(tf.ones(shape=[K,1]), K), name='sigma')

    N = tf.shape(points)[0]

    pdf = logP(points, mu, stddev)
    print tf.shape(pdf)
    pi = tf.exp(phi) / tf.reduce_sum(tf.exp(phi))
    loss = -tf.reduce_sum(reduce_logsumexp(tf.multiply(pi, pdf)))

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(loss=loss)
    return points, mu, loss, train, pdf

learning_rate = 0.01
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
        print prob[:, 0:10]

        # print prob
        # print prob.shape

        assign = np.argmax(prob, axis = 0)
        colors = [color_list[assign[i]] for i in range(len(x))]
        plt.scatter(x,y,c=colors)
        colors = ['k' for i in range(centroid.shape[0])]
        plt.scatter(centroid[:,0],centroid[:,1],c=colors)
        #plt.axis([0,500, 0, 2])
        plt.title("K mean plots for k = %d"%(K))
        plt.show()
