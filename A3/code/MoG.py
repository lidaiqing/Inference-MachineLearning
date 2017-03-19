import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *


data = np.load("../data/data2D.npy")
points_2d = data
x = points_2d[:,0]
y = points_2d[:,1]

def get_covariance_mat(sigma):

    return tf.diag_part(tf.square(sigma))

# oneDivSqrtTwoPI = 1 / math.sqrt(2*tf.math.pi)
def get_mul_normal(x, mu, sigma):
    covariance = get_covariance_mat(sigma)
    return tf.pow(2 * np.pi,-tf.shape(x)[1] / 2) * tf.pow(tf.sqrt(tf.reduce_sum(tf.square(covariance))), - 0.5) \
    * tf.exp(-0.5 * tf.transpose(x - mu) * tf.inv(covariance) * (x - mu))

def buildGraph_mog_Adam(K, learning_rate):
    # Variable creation
    points = tf.placeholder(tf.float32, [None, 2], name='input_points')
    mu = tf.Variable(tf.truncated_normal(shape=[K,2], stddev=0.5), name='mu')
    sigma = tf.Variable(tf.truncated_normal(shape=[K,2], stddev=0.5), name='sigma')
    pi =  tf.Variable(tf.div(tf.ones([K]),K), name="pi")

    N = tf.shape(points)[0]
    # Replicate to N copies of each centroid and K copies of each
    # point, then subtract and compute the sum of squared distances.
    # rep_mu = tf.reshape(tf.tile(mu, [N, 1]), [N, K, 2])
    # rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
    # sum_squares = tf.reduce_sum(tf.square(rep_points - rep_mu), reduction_indices=2)
    # best_mu = tf.argmin(sum_squares, 1)
    # count = tf.to_float(tf.unsorted_segment_sum(tf.ones_like(points), best_mu, K))
    # percentage = tf.div(count, tf.to_float(N))
    # Loss definition
    # indices_pair = tf.concat(1, [tf.reshape(tf.range(0, N), [-1,1]), tf.to_int32(tf.reshape(best_mu, [-1,1]))])
    # loss = tf.reduce_sum(tf.gather_nd(sum_squares, indices_pair))

    loss = tf.reduce_sum(reduce_logsumexp(pi * get_mul_normal(points, mu, sigma)))

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(loss=loss)
    return points, mu, loss, train

learning_rate = 0.01
points, mu, loss, train = buildGraph_mog_Adam(3, learning_rate)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
loss_recorder = np.array([])

numIteration = 500
for itr in range(numIteration):
    loss_, _ = sess.run([loss, train], feed_dict={points: points_2d})
    loss_recorder = np.append(loss_recorder, loss_)
    if itr % 100 == 0:
        print("Iteration#: %d, loss: %0.2f"%(itr, loss_))
plt.plot(np.arange(numIteration), loss_recorder, 'g')
#plt.axis([0,500, 0, 2])
plt.title("Total loss VS number of updates for learning_rate = %0.3f"%(learning_rate))
plt.show()
