from __future__ import print_function
import time
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "localhost:2222"
        ]
    })

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "10.10.1.1:2222"
        ],
    "worker" : [
        "10.10.1.1:2223",
        "10.10.1.2:2222"
        ]
    })

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "10.10.1.1:2222"
        ],
    "worker" : [
        "10.10.1.1:2223",
        "10.10.1.2:2222",
        "10.10.1.3:2222",
        "10.10.1.4:2222",
        "10.10.1.5:2222"
        ]
    })

clusterSpec = {
        "single": clusterSpec_single,
        "cluster": clusterSpec_cluster,
        "cluster2": clusterSpec_cluster2
        }

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

def static_rnn(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # downloading mnist data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Training Parameters
    learning_rate = 0.01
    training_steps = 1200
    batch_size = 64
    display_step = 200

    # Network Parameters
    num_input = 28 # MNIST data input (img shape: 28*28)
    timesteps = 28 # timesteps
    num_hidden = 128 # hidden layer num of features
    num_classes = 10 # MNIST total classes (0-9 digits)

    # test inputs
    test_x = mnist.test.images[:2000]
    test_y = mnist.test.labels[:2000]

    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Define weights
    weights = {
            'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
            }
    biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
            }

    logits = static_rnn(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the all the global variables
    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:

        # initialize session
        sess.run(init)
        time_begin = time.time()

        for step in range(training_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,\
                    Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                        "{:.4f}".format(loss) + ", Training Accuracy= " + \
                        "{:.3f}".format(acc))

        time_end = time.time()
        print("Training Finished!")

        # Calculate accuracy for 128 mnist test images
        test_len = 10000
        test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
