import numpy as np
import tensorflow as tf


# Calculate the pairwise squared Euclidean distance function for
# two input matrices (x,z)
def pair_dist(x, z):

    # Transform 1D tensor to 2D

    if len(x.get_shape()) == 1:
        x = tf.reshape(x,[1,x.get_shape().as_list()[0]])

    if len(z.get_shape()) == 1:
        z = tf.reshape(z,[1,z.get_shape().as_list()[0]])

    # Reshape x to Bx1xN
    x = tf.reshape(x,tf.pack([x.get_shape()[0],1,x.get_shape()[1]]))

    # Reshape z to 1xCxN
    z = tf.reshape(z,tf.pack([1,z.get_shape()[0],z.get_shape()[1]]))

    # Broadcasting x to BxCxN
    x = tf.tile(x,tf.pack([1,z.get_shape()[1],1]))

    # Broadcasting z to BxCxN
    z = tf.tile(z,tf.pack([x.get_shape()[0],1,1]))

    diff = tf.squared_difference(x,z)

    return tf.reduce_sum(diff,2)

def euclid_distance(X, Z):
	X = (tf.expand_dims(X, 1))
	Z = (tf.expand_dims(Z, 0))
	return tf.reduce_sum(tf.squared_difference(X, Z), 2)

def main():
    # x = tf.constant([[1,2],[3,4],[5,6]])
    # z = tf.constant([[1,2],[3,4],[5,6],[7,8]])
    x = tf.constant([[1,2],[3,4]])
    z = tf.constant([[1,1],[2,2]])



    init = tf.global_variables_initializer()
    sess = tf.Session()
    # print sess.run(x)
    # print sess.run(z)
    sess.run(init)

    print(sess.run(pair_dist(x,z)))
    # values,indices = tf.nn.top_k(pair_dist(x,z),1)
    # print(sess.run([indices,values]))


if __name__ == '__main__':
    main()
