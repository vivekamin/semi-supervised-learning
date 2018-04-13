import numpy as np
from load_dataset import mnist
import matplotlib.pyplot
import pdb
import tensorflow as tf
import math


def main():
    def kl_divergence(p, p_hat):
        return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)

    # getting the subset dataset from MNIST: 100 data of each digits

    digits = [0,0,0,0,0,0,0,0,0,0]
    index_to_get = []
    train_data, train_label, test_data, test_label = mnist(ntrain=6000, ntest=1000, digit_range=[0, 10])

    index_l = 0
    for i in range(0,train_label.shape[1]):
        #print (train_label[0][i])
        if digits[int(train_label[0][i])] == 100:
            continue
        else:
            digits[int(train_label[0][i])]+=1
            index_to_get.append(index_l)

        index_l+=1


    test_index = list(set([i for i in range(0, 6000)]) - set(index_to_get))
    #print(len(test_index))

    train_data_new = train_data.take(index_to_get,axis=1)
    train_label_new = train_label.take(index_to_get)

    test_data_new = train_data.take(test_index, axis=1)
    test_label_new = train_label.take(test_index)

    # print(train_data_new.shape, train_label_new.shape)
    # print(test_data_new.shape, test_label_new.shape)
    # print (train_label_new)

    n_values = np.max(train_label_new.astype(int)) + 1
    train_label_new = np.eye(n_values)[train_label_new.astype(int)]

    n_values = np.max(test_label_new.astype(int)) + 1
    test_label_new = np.eye(n_values)[test_label_new.astype(int)]

    print (train_label_new.shape, test_label_new.shape)

    #exit()


    learning_rate = 1e-3
    epochs = 1000
    reg_term_lambda = 1e-3
    p = 0.1
    beta = 3

    # FC Neural network to test accuracy start
    x_fc = x = tf.placeholder(tf.float32, [None, 784])
    y_label_input_fc = tf.placeholder(tf.float32, [None, 10])

    W1_fc = tf.Variable(tf.random_normal([784, 200], stddev=0.03), name='W1_FC')
    b1_fc = tf.Variable(tf.random_normal([200]), name='b1_fc')

    W2_fc = tf.Variable(tf.random_normal([200, 10], stddev=0.03), name='W2_FC')
    b2_fc = tf.Variable(tf.random_normal([10]), name='b2_fc')

    l1_linear_output = tf.add(tf.matmul(x_fc, W1_fc), b1_fc)
    l1_activation = tf.nn.relu(l1_linear_output)

    y_fc = tf.nn.softmax(tf.add(tf.matmul(l1_activation,W2_fc), b2_fc))

    cross_entropy_fc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label_input_fc, logits=y_fc))
    softmax_classifier_fc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-08).minimize(cross_entropy_fc)
    # FC Neural network to test accuracy end



    # Sparse AutoEncoder starts #
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 784])
    y_label_input = tf.placeholder(tf.float32,[None,10])

    W1 = tf.Variable(tf.random_normal([784, 200], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([200]), name='b1')

    W2 = tf.Variable(tf.random_normal([200, 784], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([784]), name='b2')

    # softmax classifier weight initialization
    W3 = tf.Variable(tf.random_normal([200, 10], stddev=0.03), name='W3')
    b3 = tf.Variable(tf.random_normal([10]), name='b3')

    linear_layer_one_output = tf.add(tf.matmul(x, W1), b1)
    layer_one_output = tf.nn.sigmoid(linear_layer_one_output)

    linear_layer_two_output = tf.add(tf.matmul(layer_one_output, W2), b2)
    y_ = tf.nn.sigmoid(linear_layer_two_output)

    # connect softmax with feature extractor
    y_label = tf.nn.softmax(tf.add(tf.matmul(layer_one_output, W3), b3))

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label_input, logits=y_label))

    softmax_classifier = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cross_entropy)

    diff = y_ - x

    p_hat = tf.reduce_mean(tf.clip_by_value(layer_one_output, 1e-10, 1.0), axis=0)

    # p_hat = tf.reduce_mean(layer_one_output,axis=1)
    kl = kl_divergence(p, p_hat)

    cost = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1)) + reg_term_lambda * (
                tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) + beta * tf.reduce_sum(kl)

    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(
        cost)

    init_op = tf.global_variables_initializer()


    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        #total_batch = int(len(mnist.train.labels) / batch_size)
        print("Training Sparse Autoencoder....")
        for epoch in range(epochs):
            #batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)

            _, c = sess.run([optimiser, cost], feed_dict={x: train_data_new.T})
            if((epoch+1)%200 == 0):
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c))


        print("Training softmax classifier with Autoencoder feature representations....")
        for epoch in range(epochs):
            cost = 0
            _ , cost = sess.run([softmax_classifier, cross_entropy], feed_dict={x: train_data_new.T, y_label_input : train_label_new})
            if ((epoch + 1) % 200 == 0):
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost))
            #print("Epoch:", (epoch + 1), "softmax classifier cost =", "{:.3f}".format(cost))


        print("Training FC network of dimensions [784, 200, 10]....")
        for epoch in range(epochs):
            cost_fc = 0
            _ , cost_fc = sess.run([softmax_classifier_fc, cross_entropy_fc], feed_dict={x_fc: train_data_new.T, y_label_input_fc : train_label_new})
            if ((epoch + 1) % 200 == 0):
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost_fc))
            #print("Epoch:", (epoch + 1), "fully connected softmax classifier cost =", "{:.3f}".format(cost_fc))


        correct_prediction = tf.equal(tf.argmax(y_label_input,1), tf.argmax(y_label,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        correct_prediction_fc = tf.equal(tf.argmax(y_label_input_fc, 1), tf.argmax(y_fc, 1))
        accuracy_fc = tf.reduce_mean(tf.cast(correct_prediction_fc, tf.float32))


        print("\nAccuracy on the unlabeled data of fully connected network is", sess.run(accuracy_fc, feed_dict={x_fc: test_data_new.T, y_label_input_fc : test_label_new}))
        print("Accuracy on the unlabeled data of Sparse Autoencoder network is", sess.run(accuracy, feed_dict={x: test_data_new.T, y_label_input : test_label_new}))


if __name__ == "__main__":
    main()