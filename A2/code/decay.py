import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *

with np.load("../data/notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]
    # reshape dataset
    trainData = trainData.reshape(trainData.shape[0],-1)
    validData = validData.reshape(validData.shape[0],-1)
    testData = testData.reshape(testData.shape[0],-1)
    # use one hot encoding for target
    trainTarget = dense_to_one_hot(trainTarget, 10)
    validTarget = dense_to_one_hot(validTarget, 10)
    testTarget = dense_to_one_hot(testTarget, 10)

# Parameters
learning_rate_list = [0.001]#, 0.01, 0.1]
batch_size = 100
display_step = 1
decay_rate = 3e-4

# Network Parameters
n_hidden_1 = 1000 # 1st layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

def buildGraph(decay_rate, learning_rate):
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,10], name='target_y')

    # Graph definition
    layer_1, W1 = layerBlock_decay(X, n_hidden_1, decay_rate)
    layer_1 = tf.nn.relu(layer_1)
    out_layer = layerBlock(layer_1, n_classes)
    y_predicted = out_layer

    # Error definition
    # print tf.get_collection('losses')
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_target, logits = y_predicted)) \
        + tf.add_n(tf.get_collection('losses'))

    correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_target,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(loss=error)
    return X, y_target, y_predicted, error, train, acc, W1

for learning_rate in learning_rate_list:
    X, y_target, y_predicted, error, train, acc, W1 = buildGraph(decay_rate, learning_rate)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    train_loss_recorder = np.array([])
    train_mis_recorder = np.array([])
    valid_loss_recorder = np.array([])
    valid_mis_recorder = np.array([])
    test_loss_recorder = np.array([])
    test_mis_recorder = np.array([])

    numIteration = 10
    for itr in range(numIteration):
        batch_xs, batch_ys = getRandomBatch(trainData, trainTarget, batch_size)
        accuracy, loss, _ = sess.run([acc, error, train], feed_dict={X: batch_xs, y_target: batch_ys})
        train_loss_recorder = np.append(train_loss_recorder, loss)
        train_mis_recorder = np.append(train_mis_recorder, 1.0 - accuracy)

        valid_loss, valid_acc = sess.run([error, acc], feed_dict={X: validData, y_target: validTarget})
        valid_loss_recorder = np.append(valid_loss_recorder, valid_loss)
        valid_mis_recorder = np.append(valid_mis_recorder, 1.0 - valid_acc)

        test_loss, test_acc = sess.run([error, acc], feed_dict={X: testData, y_target: testTarget})
        test_loss_recorder = np.append(test_loss_recorder, test_loss)
        test_mis_recorder = np.append(test_mis_recorder, 1.0 - test_acc)

        print("Iteration: %d, Train Acc: %0.3f, Valid Acc: %0.3f"%(itr, accuracy*100, valid_acc*100))

    plt.figure()
    plt.plot(np.arange(numIteration), train_loss_recorder, 'g', label="Train")
    plt.plot(np.arange(numIteration), valid_loss_recorder, 'r', label="Valid")
    plt.plot(np.arange(numIteration), test_loss_recorder, 'b', label="Test")
    plt.legend()
    #plt.axis([0,500, 0, 2])
    plt.title("Total loss VS number of updates for learning_rate = %0.3f"%(learning_rate))
    plt.show(block=False)

    plt.figure()
    plt.plot(np.arange(numIteration), train_mis_recorder, 'g', label="Train")
    plt.plot(np.arange(numIteration), valid_mis_recorder, 'r', label="Valid")
    plt.plot(np.arange(numIteration), test_mis_recorder, 'b', label="Test")
    plt.legend()
    #plt.axis([0,500, 0, 2])
    plt.title("Error VS number of updates for learning_rate = %0.3f"%(learning_rate))
    plt.show(block=False)

    plt.figure()
    f, axarr = plt.subplots(40,2)
    W1_eval = W1.eval()
    # print W1_eval.shape
    # print W1_eval[:,0].shape
    for i in range(40):
        for j in range(2):
            axarr[i,j].imshow(W1_eval[:,i*25+j].reshape(28,28), cmap='gray')
            axarr[i,j].axis('off')

    plt.show(block=False);

plt.show()
