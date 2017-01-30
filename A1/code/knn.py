import numpy as np
import tensorflow as tf
from pair_dist import pair_dist
import matplotlib.pyplot as plt

# Generate data
np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
    + 0.5 * np.random.randn(100 , 1)

randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

# Plot the data and targets
plt.plot(trainData,trainTarget,'ro',label='train')
plt.plot(validData,validTarget,'bo',label='valid')
plt.plot(testData,testTarget,'go',label='test')
plt.legend(bbox_to_anchor=(0.05, 0.95), loc=2, borderaxespad=0.)


def get_responsibility(m,k):
    # Use negtive sign to get the smallest entries in the matrix
    # _,idx = tf.nn.top_k(-m,k)
    # print type(idx)
    # print np.concatenate(np.repeat(np.arange(m.get_shape().as_list()[0]),k),idx.as_list().reshape(-1)).shape
    #
    #
    # return tf.SparseTensor(indices = np.repeat(np.arange(m.get_shape()[0]),k))
    return tf.nn.top_k(-m,k)


def get_prediction(indices,trTarget):
    # DAMN I don't know how to vectorize things here
    # so I use loops
    # print indices.get_shape()
    prediction = tf.Variable(tf.zeros(indices.get_shape().as_list()[0]))
    for i in range(indices.get_shape().as_list()[0]):
        pass



    # row = np.repeat(np.arange(indices.get_shape().as_list()[0]).reshape(-1,1),2,axis=1)


def main():

    # The only hyperparameter in KNN
    k = 3

    tsData = tf.placeholder(tf.float32,shape = (testData.shape))
    trData = tf.placeholder(tf.float32,shape = (trainData.shape))
    vaData = tf.placeholder("float")
    tsTarget = tf.placeholder("float")
    trTarget = tf.placeholder("float")
    vaTarget = tf.placeholder("float")

    pdist = pair_dist(tsData,trData)
    _,indices = get_responsibility(pdist,k)
    # get_prediction(indices,trTarget)

    with tf.Session() as sess:
        print sess.run(indices,feed_dict={tsData:testData,trData:trainData,vaData:validData,\
            tsTarget:testTarget,trTarget:trainTarget,vaTarget:validTarget})

if __name__ == '__main__':
    main()
    plt.show()
