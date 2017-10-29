import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)


# A is a 0-dimensional int32 tensor
A = tf.constant(1234)
# B is a 1-dimensional int32 tensor
B = tf.constant([123, 456, 789])
# C is a 2-dimensional int32 tensor
C = tf.constant([[123, 456, 789], [222, 333, 444]])

with tf.Session() as sess:
    output_A = sess.run(A)
    output_B = sess.run(B)
    output_C = sess.run(C)
    print('\nA is ', output_A, '\nB is ', output_B, '\nC is ', output_C)

# Session feed dict
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
    print(output)


with tf.Session() as sess:
    output = sess.run([x, y, z],
                      feed_dict={x: 'Test String', y: 123, z: 45.67})
    print(output)

x1 = tf.add(5, 2)
x2 = tf.subtract(10, 4)
x3 = tf.multiply(2, 5)

with tf.Session() as sess:
    output = sess.run([x1, x2, x3])
    output2 = sess.run(x2)
    print(output)
    print(output2)


x1 = tf.constant(1.0)
x2 = tf.constant(2)
x3 = tf.add(x1, x2)
x4 = tf.add(x1, tf.cast(x2, tf.float32))
with tf.Session() as sess:
    # This will fail due to different type
    try:
        sess.run(x3)
    except TypeError:
        print("type is not the same")
    # This is successful
    sess.run(x4)

# Variable is slightly different to constants where we need to initialise it.
x = tf.Variable(10)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    output = x.eval()
    print(output)


n_features = 120
n_labels = 5
n = 10
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
bias = tf.Variable(tf.zeros(n_labels))
x = tf.Variable(tf.truncated_normal((n, n_features)))
y_hat = tf.add(tf.matmul(x, weights), bias)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    w = weights.eval()
    b = bias.eval()
    print(w.shape)
    print(b.shape)
    y = sess.run(y_hat)
    print(y)


def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)

    # TODO: Calculate the softmax of the logits
    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        # TODO: Feed in the logit data
        # output = sess.run(softmax,    )
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output


run()


x = tf.constant([1, 2, 3, 4, 5])
y = tf.reduce_sum(x)
z = tf.log(tf.cast(y, tf.float32))
with tf.Session() as sess:
    output = sess.run(z)
    print(output)

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]
softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
ce = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))
with tf.Session() as sess:
    output = sess.run(
        ce, feed_dict={softmax: softmax_data, one_hot: one_hot_data})
    print(output)

# Using relu for a 2 layer network
output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable(
    [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model
h1i = tf.add(tf.matmul(features, weights[0]), biases[0])
h1o = tf.nn.relu(h1i)
h2i = tf.add(tf.matmul(h1o, weights[1]), biases[1])


# TODO: Print session results
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(h2i)
    print(output)
