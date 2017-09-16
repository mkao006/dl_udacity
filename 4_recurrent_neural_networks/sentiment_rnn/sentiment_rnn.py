from __future__ import division
import numpy as np
import tensorflow as tf
from collections import Counter
from string import punctuation
from sklearn.model_selection import train_test_split

with open('reviews.txt', 'r') as f:
    reviews = f.read()
with open('labels.txt', 'r') as f:
    labels = f.read()


def text_preprocessing(text):
    ''' Function to process the text.

    1. Punctuation are removed.
    2. sentences are splitted to give tokens.
    '''
    punctuation_removed = ''.join([t for t in text if t not in punctuation])
    text_list = punctuation_removed.split('\n')
    tokens = ' '.join(text_list).split()
    return text_list, tokens


def create_vocab_to_int_dict(tokens):
    ''' Function to create dictionary to convert token into index.
    '''

    token_set = list(set(tokens))
    conversion_dict = {t: i for i, t in enumerate(token_set)}
    return conversion_dict


def sentiment_preprocessing(labels):
    ''' Function to process labels.

    Text is splitted and then converted to binary labels.
    '''

    splitted_labels = labels.split('\n')
    binary_labels = np.array([1 if each == 'positive' else 0
                              for each in splitted_labels])
    return binary_labels


def create_feature_matrix(text_list, max_length=200):
    ''' Function to convert the list of reviews in to a feature matrix.

    padding and truncation is performed to the specified length.
    '''
    feature_matrix = np.zeros(
        (len(text_list), max_length), dtype=int)
    for i, l in enumerate(text_list):
        feature_matrix[i, -len(l):] = (
            np.array(text_list[i])[:max_length])
    return feature_matrix


def create_data_partition(feature, labels, train_frac=0.8, test_frac=0.1,
                          verbose=True):
    ''' Wrapper function to create the train, validation and test data sets.
    '''

    train_x, val_test_x, train_y, val_test_y = (
        train_test_split(features, labels, train_size=train_frac))

    remain_test_frac = test_frac / (1 - train_frac)
    test_x, val_x, test_y, val_y = (
        train_test_split(val_test_x, val_test_y, train_size=remain_test_frac))

    if verbose:
        print('\nTrain shape: {}'.format(train_x.shape))
        print('Test shape: {}'.format(test_x.shape))
        print('Validation shape: {}'.format(val_x.shape))

    return train_x, train_y, test_x, test_y, val_x, val_y


def get_batches(x, y, batch_size):
    ''' Function to create generator function to generate batches.
    '''

    n_batches = len(x) // batch_size
    batch_sample = n_batches * batch_size
    batch_x = x[:batch_sample]
    batch_y = y[:batch_sample]
    for start in range(0, batch_sample, batch_size):
        end = start + batch_size
        yield batch_x[start:end], batch_y[start:end]


class SentimentRnn:
    ''' Class to model the sentiments with RNN
    '''

    def __init__(self,
                 vocab_size,
                 lstm_size=256,
                 lstm_layers=1,
                 batch_size=500,
                 learning_rate=0.001,
                 embed_size=300,
                 epochs=10):
        ''' Initialise the graph.
        '''

        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embde_size = embed_size
        self.epochs = epochs

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create inputs
            self.inputs_ = tf.placeholder(
                tf.int32, [None, None], name='inputs')
            self.labels_ = tf.placeholder(
                tf.int32, [None, None], name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # Create embedding
            embedding = tf.Variable(
                tf.random_uniform(shape=(vocab_size, embed_size),
                                  minval=-1, maxval=1))
            embed = tf.nn.embedding_lookup(params=embedding, ids=self.inputs_)

            # Define the RNN
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size)
            drop = tf.contrib.rnn.DropoutWrapper(
                cell=cell, output_keep_prob=self.keep_prob)
            self.lstm = tf.contrib.rnn.MultiRNNCell(cells=[drop] * lstm_layers)
            self.initial_state = self.lstm.zero_state(
                batch_size=batch_size, dtype=tf.float32)
            outputs, self.final_state = (
                tf.nn.dynamic_rnn(cell=self.lstm,
                                  inputs=embed,
                                  initial_state=self.initial_state))

            # Create prediction
            predictions = (
                tf.contrib.layers.fully_connected(inputs=outputs[:, -1],
                                                  num_outputs=1,
                                                  activation_fn=tf.sigmoid))

            # Training parameters
            self.cost = tf.losses.mean_squared_error(labels=self.labels_,
                                                     predictions=predictions)
            self.optimiser = (
                tf.train.AdamOptimizer(learning_rate=learning_rate)
                .minimize(loss=self.cost))

            # Validation
            predicted_class = tf.cast(x=tf.round(predictions), dtype=tf.int32)
            correct_pred = tf.cast(x=tf.equal(predicted_class, self.labels_),
                                   dtype=tf.float32)
            self.accuracy = tf.reduce_mean(input_tensor=correct_pred)

    def train(self, train_x, train_y, val_x, val_y):
        ''' Train the RNN
        '''
        with self.graph.as_default():
            self.saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            iteration = 1
            for e in range(self.epochs):
                state = sess.run(self.initial_state)

                for ii, (x, y) in enumerate(get_batches(train_x, train_y,
                                                        self.batch_size),
                                            start=1):
                    feed = {self.inputs_: x,
                            self.labels_: y[:, None],
                            self.keep_prob: 0.5,
                            self.initial_state: state}
                    loss, state, _ = sess.run([self.cost,
                                               self.final_state,
                                               self.optimiser],
                                              feed_dict=feed)

                    if iteration % 5 == 0:
                        print("Epoch: {}/{}".format(e, self.epochs),
                              "Iteration: {}".format(iteration),
                              "Train loss: {:.3f}".format(loss))

                    if iteration % 25 == 0:
                        val_acc = []
                        val_state = sess.run(
                            self.lstm.zero_state(self.batch_size, tf.float32))
                        for x, y in get_batches(val_x, val_y, self.batch_size):
                            feed = {self.inputs_: x,
                                    self.labels_: y[:, None],
                                    self.keep_prob: 1,
                                    self.initial_state: val_state}
                            batch_acc, val_state = sess.run([self.accuracy,
                                                             self.final_state],
                                                            feed_dict=feed)
                            val_acc.append(batch_acc)
                        print("Val acc: {:.3f}".format(np.mean(val_acc)))
                    iteration += 1

            self.saver.save(sess, "checkpoints/sentiment.ckpt")

    def test(self, test_x, test_y):
        ''' Test the RNN with test data.
        '''
        test_acc = []
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            test_state = sess.run(self.lstm.zero_state(
                self.batch_size, tf.float32))
            for ii, (x, y) in enumerate(get_batches(test_x, test_y,
                                                    self.batch_size), 1):
                feed = {self.inputs_: x,
                        self.labels_: y[:, None],
                        self.keep_prob: 1,
                        self.initial_state: test_state}
                batch_acc, test_state = sess.run(
                    [self.accuracy, self.final_state], feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


# Preprocess the data
review_list, words = text_preprocessing(reviews)


# Create your dictionary that maps vocab words to integers here
vocab_to_int = create_vocab_to_int_dict(words)

# Convert the reviews to integers, same shape as reviews list, but with
# integers
reviews_ints = [[vocab_to_int[word] for word in review.split()]
                for review in review_list]


# Remove zero index
non_zero_ind = [i for i, r in enumerate(reviews_ints) if len(r) > 0]
reviews_ints = [reviews_ints[i] for i in non_zero_ind]

# Convert labels to 1s and 0s for 'positive' and 'negative'
labels = sentiment_preprocessing(labels)
labels = labels[non_zero_ind]


# create feature matrix
features = create_feature_matrix(reviews_ints)


review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# Create the data partition
train_x, train_y, test_x, test_y, val_x, val_y = (
    create_data_partition(features, labels))

# Define the vocab size
vocab_size = len(vocab_to_int)

# Train the model
srnn = SentimentRnn(vocab_size=vocab_size)
srnn.train(train_x, train_y, val_x, val_y)
srnn.test(test_x, test_y)
