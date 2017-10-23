from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



# Network Parameters
feature_hidden_1 = 256 # 1st layer features
feature_hidden_2 = 128 # 2nd layer features
#feature_hidden_3 = 64 # 3nd layer features
feature_input = 784 # MNIST image feature set is 28 X 28

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, feature_input])

weights_layers = {
    'encoder_h1': tf.Variable(tf.random_normal([feature_input, feature_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([feature_hidden_1, feature_hidden_2])),
    #'encoder_h3': tf.Variable(tf.random_normal([feature_hidden_2, feature_hidden_3])),
    #'decoder_h1': tf.Variable(tf.random_normal([feature_hidden_3, feature_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([feature_hidden_2, feature_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([feature_hidden_1, feature_input])),
}
biases_layers = {
    'encoder_b1': tf.Variable(tf.random_normal([feature_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([feature_hidden_2])),
    #'encoder_b3': tf.Variable(tf.random_normal([feature_hidden_3])),
    #'decoder_b1': tf.Variable(tf.random_normal([feature_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([feature_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([feature_input])),
}

# sigmoid encoder with layers
def encoder_sigmoid(x):
    # Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_layers['encoder_h1']), biases_layers['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_layers['encoder_h2']), biases_layers['encoder_b2']))
    # Encoder Hidden layer with sigmoid activation #3
    #layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights_layers['encoder_h3']), biases_layers['encoder_b3']))

    return layer_2

# sigmoid decoder with layers
def decoder_sigmoid(x):
    # Hidden layer with sigmoid activation #1
    #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_layers['decoder_h1']), biases_layers['decoder_b1']))
    # Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_layers['decoder_h2']), biases_layers['decoder_b2']))
    # Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights_layers['decoder_h3']), biases_layers['decoder_b3']))
    return layer_3

# tanh encoder with layers
def encoder_tanh(x):
    # Hidden layer with tanh activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights_layers['encoder_h1']), biases_layers['encoder_b1']))
    # Hidden layer with tanh activation #2
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights_layers['encoder_h2']), biases_layers['encoder_b2']))
    # Hidden layer with tanh activation #3
    #layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights_layers['encoder_h3']), biases_layers['encoder_b3']))
    return layer_2

#tanh decoder with layers
def decoder_tanh(x):
    # Hidden layer with tanh activation #1
    #layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights_layers['decoder_h1']), biases_layers['decoder_b1']))
    # Hidden layer with tanh activation #1
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(x, weights_layers['decoder_h2']), biases_layers['decoder_b2']))
    # Hidden layer with tanh activation #1
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights_layers['decoder_h3']), biases_layers['decoder_b3']))
    return layer_3

# Construct model
encoder_op = encoder_sigmoid(X)
# put encoder result in decoder operation
decoder_op = decoder_sigmoid(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X


# Training Parameters
learning_rate = 0.01
num_steps = 32000
batch_size = 1024

display_step = 4000
examples_to_show = 10


# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 10
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()

""" Result
CPU: 
GPU : GeForce GTX 1070
major: 6 minor: 1 memoryClockRate (GHz) 1.7465
Total memory: 8.00GiB


Experiment 1:
hidden layers: 3
learning_rate = 0.01
num_steps = 32000
batch_size = 1024
display_step = 4000

with Tanh function:
Step 1: Minibatch Loss: 1.063042
Step 4000: Minibatch Loss: 0.833495
Step 8000: Minibatch Loss: 0.859443
Step 12000: Minibatch Loss: 0.860573
Step 16000: Minibatch Loss: 0.862071
Step 20000: Minibatch Loss: 0.858836
Step 24000: Minibatch Loss: 0.858948
Step 28000: Minibatch Loss: 0.861257
Step 32000: Minibatch Loss: 0.857677

With Sigmoid function:
Step 1: Minibatch Loss: 0.450492
Step 4000: Minibatch Loss: 0.095716
Step 8000: Minibatch Loss: 0.083560
Step 12000: Minibatch Loss: 0.068244
Step 16000: Minibatch Loss: 0.064311
Step 20000: Minibatch Loss: 0.062675
Step 24000: Minibatch Loss: 0.059845
Step 28000: Minibatch Loss: 0.056680
Step 32000: Minibatch Loss: 0.057008

Experiment 2:
Hidden Layers: 2
learning_rate = 0.01
num_steps = 32000
batch_size = 1024
display_step = 4000
with Tanh function:
Step 1: Minibatch Loss: 1.059582
Step 4000: Minibatch Loss: 0.822155
Step 8000: Minibatch Loss: 0.837025
Step 12000: Minibatch Loss: 0.830645
Step 16000: Minibatch Loss: 0.839630
Step 20000: Minibatch Loss: 0.833946
Step 24000: Minibatch Loss: 0.835188
Step 28000: Minibatch Loss: 0.835814
Step 32000: Minibatch Loss: 0.840790

With Sigmoid function:
Step 1: Minibatch Loss: 0.441761
Step 4000: Minibatch Loss: 0.080552
Step 8000: Minibatch Loss: 0.069778
Step 12000: Minibatch Loss: 0.063971
Step 16000: Minibatch Loss: 0.058334
Step 20000: Minibatch Loss: 0.052234
Step 24000: Minibatch Loss: 0.050157
Step 28000: Minibatch Loss: 0.046284
Step 32000: Minibatch Loss: 0.042556




"""