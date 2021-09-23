import tensorflow as tf
import sys
import numpy as np
import random

if sys.argv[1] is not None:
    model_name = sys.argv[1]

num_run = sys.argv[2]

n_hidden = 25
n_output = 1
nMax = 15
learning_rate = 1e-2
nImages_train_total = 14993  # constfN: 15000, decrsfN: 14993
nImages_test = 1000
batch_size = 100

training_epochs = 10001  # 1000001
num_test_epochs = 1
checkpoints = [10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000,
               10000, 15000, 20000, 25000, 50000]  # , 500000, 1000000]

save_dir = "scalar_model_sd0.3_decrsfN_new/"+model_name
restore = False
start_restore_index = 0
load_file = save_dir+"/scalar_model_data_run"+str(num_run)+"_epoch"+str(start_restore_index)+".ckpt"

all_indices = [x for x in range(0, 25)]


def generate_training():
    dataset = []

    S = 0.0
    for i in range(1, nMax+1):
        S += i ** (-2)

    for num in range(1, nMax+1):
        #nImages_train = 1000 # constfN
        nImages_train = int(15000*(num**(-2))/S) # decrsfN
        for img in range(nImages_train):
            stimulus = np.zeros(25)
            label = np.zeros(15)
            indices = random.sample(all_indices, num)
            for index in indices:
                stimulus[index] = np.random.normal(1.0, 0.3, 1)[0]
            label[num-1] = 1.0
            dataset.append([stimulus, label])
    random.shuffle(dataset)

    x_train = []
    y_train = []

    for tup in dataset:
        x_train.append(tup[0])
        y_train.append(tup[1])
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return x_train, y_train


def generate_testing():
    dataset = []
    for num in range(1, nMax+1):
        for img in range(nImages_test):
            stimulus = np.zeros(25)
            label = np.zeros(15)
            indices = random.sample(all_indices, num)
            for index in indices:
                stimulus[index] = np.random.normal(1.0, 0.3, 1)[0]
            label[num-1] = 1.0
            dataset.append([stimulus, label])
    random.shuffle(dataset)
    x_test = []
    y_test = []
    for tup in dataset:
        x_test.append(tup[0])
        y_test.append(tup[1])
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_test, y_test


def next_batch(images, labels, batch_size):
    """Returns a batch of size batch_size of data."""
    all_idx = np.arange(0, nImages_train_total)
    np.random.shuffle(all_idx)
    batch_idx = all_idx[:batch_size]
    batch_imgs = [images[i] for i in batch_idx]
    batch_lbls = [labels[i] for i in batch_idx]
    return batch_imgs, batch_lbls


if __name__ == '__main__':

    def output(x):
        output_layer = tf.add(tf.matmul(x, weights['output_w']), biases['output_b'])
        return output_layer

    # tf Graph
    X = tf.placeholder(tf.float32, shape=[batch_size, n_hidden])
    Y = tf.placeholder(tf.float32, shape=[batch_size, nMax])
    X_test = tf.placeholder(tf.float32, shape=[nImages_test*nMax, n_hidden])
    weights = {
        'output_w': tf.Variable(tf.random_uniform([n_hidden, n_output])),
    }

    biases = {
        'output_b': tf.Variable(tf.random_uniform([n_output])),
    }

    # Construct model
    output_op = output(X)

    # Prediction
    y_pred = output_op
    y_true = tf.cast(tf.argmax(Y, 1) + 1, tf.float32)
    y_true = tf.reshape(y_true, [-1, n_output])
    y_pred_test = output(X_test)

    ### LOSE FUNCTION ###
    predquality = tf.square(tf.subtract(y_pred, y_true))
    tf.cast(predquality, tf.float32)
    cost = tf.reduce_mean(predquality)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1).minimize(cost)

    init = tf.global_variables_initializer()

    # Launch the graph
    # Using InteractiveSession (more convenient while using Notebooks)
    sess = tf.InteractiveSession()

    # Initializing the variables
    sess.run(init)

    if restore:
        saver.restore(sess, load_file)

    # Training cycle
    for train_epoch in range(start_restore_index, training_epochs):
        x_train, y_train = generate_training()
        x_train_data, y_train_data = next_batch(x_train, y_train, batch_size)
        pred, _, pred_q, c = sess.run([y_pred, optimizer, predquality, cost], feed_dict={X: x_train_data, Y: y_train_data})

        # Display logs per epoch step
        if train_epoch in checkpoints:
            print("Epoch:", '%04d' % (train_epoch+1), "cost=", "{:.9f}".format(c))
            model_data = {}
            all_classifications = []
            all_labels = []

            w, b = sess.run([weights, biases], feed_dict={X: np.zeros((batch_size, n_hidden)), Y: np.zeros((batch_size, nMax))})

            for test_epoch in range(num_test_epochs):
                x_test_data, y_test_data = generate_testing()
                classifications = sess.run(y_pred_test, feed_dict={X_test: x_test_data})
                if len(all_classifications) == 0:
                    all_classifications = classifications
                    all_labels = y_test_data
                else:
                    all_classifications = np.concatenate((all_classifications, classifications), axis=0)
                    all_labels = np.concatenate((all_labels, y_test_data), axis=0)

            model_data["classification"] = all_classifications
            model_data["label"] = all_labels
            model_data["w"] = w
            model_data["b"] = b

            saver = tf.train.Saver(tf.global_variables())
            print("Model saved in file: %s" % saver.save(sess, save_dir+"/scalar_model_data_run"+str(num_run)+"_epoch"+str(train_epoch)+".ckpt"))
            np.save(save_dir+"/scalar_model_data_run"+str(num_run)+"_epoch"+str(train_epoch)+".npy", model_data)
