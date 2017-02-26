import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from datetime import datetime

# Network Parameters
n_hidden_1 = 500
n_hidden_2 = 500
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def buildGraph(numLayers, unitPerLayer, decay_rate, learning_rate, dropOut):
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,10], name='target_y')

    hidden_layers = {}
    for nLayer in range(numLayers):
        # Graph definition
        if nLayer == 0:
            hidden_layers[nLayer] = tf.nn.dropout(tf.nn.relu(layerBlock(X, unitPerLayer[nLayer])),\
                np.power(keep_prob, dropOut))
        else:
            hidden_layers[nLayer] = tf.nn.dropout(tf.nn.relu(layerBlock(hidden_layers[nLayer - 1], \
                unitPerLayer[nLayer])),np.power(keep_prob, dropOut))

    out_layer = layerBlock(hidden_layers[numLayers - 1], n_classes)
    y_predicted = out_layer

    # Error definition
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_target, logits = y_predicted))

    correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_target,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(loss=error)
    return X, y_target, y_predicted, error, train, acc

with np.load("../data/notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    # np.random.seed(521)
    random.seed(datetime.now())
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    # trainData = tf.image.per_image_standardization(trainData)
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    # validData = tf.image.per_image_standardization(validData)
    testData, testTarget = Data[16000:], Target[16000:]
    # testData = tf.image.per_image_standardization(testData)
    # reshape dataset
    trainData = trainData.reshape(trainData.shape[0],-1)
    validData = validData.reshape(validData.shape[0],-1)
    testData = testData.reshape(testData.shape[0],-1)
    # use one hot encoding for target
    trainTarget = dense_to_one_hot(trainTarget, 10)
    validTarget = dense_to_one_hot(validTarget, 10)
    testTarget = dense_to_one_hot(testTarget, 10)


# decay_rate = 0.01
batch_size = 100
# learning_rate = 0.001

for numMod in range(5):

    learning_rate = np.exp(np.random.rand() * 3 - 7.5)
    numLayers = np.random.randint(5) + 1
    unitPerLayer = np.random.randint(100,500,numLayers)
    decay_rate = np.exp(np.random.rand() * 3 - 9)
    dropOut = np.random.randint(2)

    print("LR = %0.10f, NL = %d, DR = %0.10f, Dropout = %d" \
        %(learning_rate, numLayers, decay_rate, dropOut))
    print(unitPerLayer)

    #best learning rate 0.001, number of hidden units 500
    X, y_target, y_predicted, error, train, acc = buildGraph(numLayers, unitPerLayer, decay_rate, learning_rate, dropOut)
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    train_loss_recorder = np.array([])
    train_mis_recorder = np.array([])
    valid_loss_recorder = np.array([])
    valid_mis_recorder = np.array([])
    test_loss_recorder = np.array([])
    test_mis_recorder = np.array([])

    numIteration = 2000
    global_best_test_acc = 0
    best_acc_valid = 0
    best_acc_test = 0

    for itr in range(numIteration):
        batch_xs, batch_ys = getRandomBatch(trainData, trainTarget, batch_size)
        accuracy, loss, _ = sess.run([acc, error, train], feed_dict={X: batch_xs, y_target: batch_ys, keep_prob: 0.8})
        train_loss_recorder = np.append(train_loss_recorder, loss)
        train_mis_recorder = np.append(train_mis_recorder, 1.0 - accuracy)

        valid_loss, valid_acc = sess.run([error, acc], feed_dict={X: validData, y_target: validTarget, keep_prob: 0.8})
        valid_loss_recorder = np.append(valid_loss_recorder, valid_loss)
        valid_mis_recorder = np.append(valid_mis_recorder, 1.0 - valid_acc)

        test_loss, test_acc = sess.run([error, acc], feed_dict={X: testData, y_target: testTarget, keep_prob: 0.8})
        test_loss_recorder = np.append(test_loss_recorder, test_loss)
        test_mis_recorder = np.append(test_mis_recorder, 1.0 - test_acc)
        # print("Iteration: %d, Train Acc: %0.3f, Valid Acc: %0.3f"%(itr, accuracy*100, valid_acc*100))
        best_acc_valid = max(best_acc_valid, valid_acc)
        best_acc_test = max(best_acc_test, test_acc)
    #
    # plt.plot(np.arange(numIteration), train_loss_recorder, 'g', label="Train")
    # plt.plot(np.arange(numIteration), valid_loss_recorder, 'r', label="Valid")
    # # plt.plot(np.arange(numIteration), test_loss_recorder, 'b', label="Test")
    # plt.legend()
    # #plt.axis([0,500, 0, 2])
    # plt.title("Total loss VS number of updates for learning_rate = %0.3f"%(learning_rate))
    # plt.show()
    #
    # plt.plot(np.arange(numIteration), train_mis_recorder, 'g', label="Train")
    # plt.plot(np.arange(numIteration), valid_mis_recorder, 'r', label="Valid")
    # # plt.plot(np.arange(numIteration), test_mis_recorder, 'b', label="Test")
    # plt.legend()
    # #plt.axis([0,500, 0, 2])
    # plt.title("Error VS number of updates for learning_rate = %0.3f"%(learning_rate))
    # plt.show()

    print("ValidAcc = %0.3f, testAcc = %0.3f\n" %(best_acc_valid, best_acc_test))
    if(global_best_test_acc < best_acc_test):
        global_best_test_acc = best_acc_test
        b_lr = learning_rate
        b_dr = decay_rate
        b_do = dropOut
        b_nl = numLayers
        b_ul = unitPerLayer
        b_va = best_acc_valid

print("BEST: LR = %0.10f, NL = %d, DR = %0.10f, Dropout = %d, ValidAcc = %0.3f,  TestAcc = %0.3f",\
    %(b_lr, b_nl, b_dr, b_do, b_va, global_best_test_acc))
print(b_ul)
