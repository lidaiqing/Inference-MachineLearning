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

    x = tf.placeholder(tf.float32, [None, 64])
    W = tf.Variable(tf.zeros([64, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, 1])

    decay_rate = 1
    learning_rate = 0.2
    bach_size = 50

    loss = tf.reduce_sum(tf.square(y_-y))/(2*trainData.shape[0]) + \
        decay_rate/2*tf.reduce_sum(tf.square(W))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(y_, (tf.sign(y-0.5)+1)/2)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    acc_recorder = np.array([])

    # Train
    for itr in range(2000):
      batch_xs, batch_ys = getRandomBatch(trainData,trainTarget,bach_size)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

      if itr %10 == 0:
          acc = sess.run(accuracy, feed_dict={x: validData,
                                          y_: validTarget})
          print "Iteration: {}, accuracy on validation data: {}".format(itr,acc)
          acc_recorder = np.append(acc_recorder,acc)

    # Test trained model
    correct_prediction = tf.equal(y_, (tf.sign(y-0.5)+1)/2)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={x: testData,
                                        y_: testTarget})
    print "Accuracy on test data: {}".format(acc)


    xaxis = np.arange(acc_recorder.shape[0])*10
    plt.plot(xaxis,acc_recorder)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on Validation Data with " + r'$\eta = {}$'.format(learning_rate))
    # plt.show()
    plt.savefig('../figures/three/700_02.png')

if __name__ == '__main__':
    main()
