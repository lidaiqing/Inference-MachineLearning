import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def getRandomBatch(trainData,trainTarget,size):
    idx = np.random.choice(trainData.shape[0], size, replace=False)
    return trainData[idx,:], trainTarget[idx,:]

def main():
    with np.load ("../data/tinymnist.npz") as data :
        trainData, trainTarget = data ["x"], data["y"]
        validData, validTarget = data ["x_valid"], data ["y_valid"]
        testData, testTarget = data ["x_test"], data ["y_test"]


    # print trainData.shape
    # print validData.shape
    # print testData.shape
    # print validTarget


    # Create the model
    x = tf.placeholder(tf.float32, [None, 64])
    W = tf.Variable(tf.zeros([64, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    # cross_entropy = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    decay_rate = 1

    loss = tf.reduce_sum(tf.square(y_-y))/(2*trainData.shape[0]) + \
        decay_rate/2*tf.reduce_sum(tf.square(W))
    # print loss
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


    correct_prediction = tf.equal(y_, (tf.sign(y-0.5)+1)/2)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    acc_recorder = np.array([])

    # Train
    for itr in range(1000):
      batch_xs, batch_ys = getRandomBatch(trainData,trainTarget,50)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

      if itr %10 == 0:
          acc = sess.run(accuracy, feed_dict={x: validData,
                                          y_: validTarget})
          print "Iteration: {}, accuracy on validation data: {}".format(itr,acc)
          acc_recorder = np.append(acc_recorder,acc)

    xaxis = np.arange(100)*10
    plt.plot(xaxis,acc_recorder)

    # Test trained model
    correct_prediction = tf.equal(y_, (tf.sign(y-0.5)+1)/2)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={x: testData,
                                        y_: testTarget})
    print "Accuracy on test data: {}".format(acc)

if __name__ == '__main__':
    main()
    plt.show()
