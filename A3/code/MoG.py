import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *

points_100d = np.load("../data/data100D.npy")
Number_of_data = points_100d.shape[0]
validSize = int(Number_of_data * 1.0 / 3.0)
validData = points_100d[:validSize,:]
trainData = points_100d[validSize:,:]

def get_log_gaussian(X, mu, diag_stdev):
    '''
    X: BxD
    mu: KxD
    diag_stdev: KxD
    B: number of data points
    D: dimension of a data point
    K: number of clusters
    return BxK matrix
    '''
    B = tf.shape(X)[0]
    D = tf.shape(X)[1]
    K = tf.shape(mu)[0]
    rep_X = tf.reshape(tf.tile(X, [1, K]), [B, K, D])
    dist = tf.contrib.distributions.MultivariateNormalDiag(mu, diag_stdev)
    return dist.log_pdf(rep_X)
    
def get_log_P(X, pi, mu, diag_stdev):
    '''
    X: BxD
    pi: 1xK
    mu: KxD
    diag_stdev: Kx1
    B: number of data points
    D: dimension of a data point
    K: number of clusters
    return float64
    '''
    B = tf.shape(X)[0]
    D = tf.shape(X)[1]
    K = tf.shape(mu)[0]
    rep_pi = tf.reshape(tf.tile(pi, [1,B]), [B, K])
    log_gaussian = get_log_gaussian(X, mu, diag_stdev)
    res = tf.reduce_sum(reduce_logsumexp(rep_pi + log_gaussian))
    return res

def get_MoG_assign(X, pi, mi, diag_stdev):
    assign = tf.argmax(get_log_P_Z_given_X(X, pi, mu, diag_stdev), 1)
    return assign

def buildGraph_MoG_Adam_100d(K, learning_rate):
    # Variable creation
    points = tf.placeholder(tf.float32, [None, 100], name='input_points')
    mu = tf.Variable(tf.truncated_normal(shape=[K,100], stddev=0.5), name='mu')

#     diag_stdev = tf.Variable(tf.exp(tf.truncated_normal(shape=[K,100], stddev=0.5)), name='diag_stdev')

#     Covariance matrix
#     diag_stdev = tf.Variable(tf.exp(tf.truncated_normal(shape=[K,100], stddev=0.5)), name='diag_stdev')
#     print "Using covariance matrix"

#     Single sigma value
    sigma = tf.Variable(tf.exp(tf.truncated_normal(shape=[K], stddev=0.5)), name='sigma')
    diag_stdev = tf.transpose(tf.reshape(tf.tile(sigma, [100], name='diag_stdev'),[100,K]))
    print "Using single sigma"

    phi = tf.Variable(tf.truncated_normal(shape=[1,K], stddev=0.5), name='phi')
    pi = logsoftmax(phi)

    # Loss definition
    loss = -get_log_P(points, pi, mu, diag_stdev)
    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(loss=loss)
    return points, pi, mu, diag_stdev, loss, train

K_list = [5,10,15,20,25]
learning_rate = 0.01
color_list = ["g", "b", "r", "y", "k"]
number_of_valid_data = validData.shape[0]
loss_recorder = np.array([])

for K in K_list:
    tf.reset_default_graph()
    points, pi, mu, diag_stdev, loss, train = buildGraph_MoG_Adam_100d(K, learning_rate)
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    numIteration = 600
    pi_, mu_, diag_stdev_ = [], [], []
    for itr in range(numIteration):
        pi_,mu_,diag_stdev_,_ = sess.run([pi, mu, diag_stdev, train], feed_dict={points: trainData})
        loss_ = sess.run(loss, feed_dict={points: validData})
        if itr % 100 == 0:
            print("Number of cluster: %d, Iteration#: %d, Validation loss: %0.2f"%(K, itr, loss_))
    loss_ = sess.run(loss, feed_dict={points: validData})
    loss_recorder = np.append(loss_recorder, loss_)


plt.plot(K_list, loss_recorder, 'g')
    #plt.axis([0,500, 0, 2])
plt.title("Total loss of validation set VS number of clusters K")
plt.show()
