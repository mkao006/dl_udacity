import time
import numpy as np
import tensorflow as tf
import zipfile
import utils
import random
from collections import Counter
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_data():
    ''' Download the claened up wikipedia articles.
    '''

    dataset_folder_path = 'data'
    dataset_filename = 'text8.zip'
    dataset_name = 'Text8 Dataset'

    if not isfile(dataset_filename):
        with DLProgress(unit='B', unit_scale=True, miniters=1,
                        desc=dataset_name) as pbar:
            urlretrieve(
                'http://mattmahoney.net/dc/text8.zip',
                dataset_filename,
                pbar.hook)

    if not isdir(dataset_folder_path):
        with zipfile.ZipFile(dataset_filename) as zip_ref:
            zip_ref.extractall(dataset_folder_path)


def load_data(path='data/text8'):
    ''' Function to load local data and create conversion tables.

    '''

    with open(path) as f:
        text = f.read()

    words = utils.preprocess(text)
    print("Total words: {}".format(len(words)))
    print("Unique words: {}".format(len(set(words))))
    vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]
    return vocab_to_int, int_to_vocab, int_words


def subsampling(word_list, threshold):
    '''This function performs subsampling as termed by Mikolov. The
    purpose of the procedure is to reduce noises caused by frequeny,
    stop words or articles such as 'the'.

    '''
    word_count = Counter(word_list)
    total_words = len(word_list)
    word_freqs = {word: count / total_words
                  for word, count in word_count.items()}

    word_prob_threshold = {word: 1 - np.sqrt(threshold / word_freqs[word])
                           for word in word_count}

    sample = [word
              for word in word_list
              if (1 - word_prob_threshold[word]) > np.random.random()]

    return sample


def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    rad = np.random.randint(1, window_size + 1)
    start = idx - rad if idx > rad else 0
    end = idx + rad
    target = list(set(words[start:idx] + words[(idx + 1):(end + 1)]))
    return target


def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    word_length = len(words)
    n_batches = word_length // batch_size
    words = words[:n_batches * batch_size]

    # Loop through the bathces
    for idx in range(0, word_length, batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y


class WordVectorizer:
    ''' A simple vectorizer to create word embedding based on word2vec.
    '''

    def __init__(self, words, n_embedding=200, n_sampled=100,
                 valid_size=16, valid_window=100):
        tf.reset_default_graph()
        n_vocab = len(words)
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input and label placeholders
            self.inputs = tf.placeholder(tf.int32, [None], name='inputs')
            self.labels = tf.placeholder(
                tf.int32, [None, None], name='labels')

            # Create embedding
            self.embedding = tf.Variable(tf.random_uniform(
                shape=(n_vocab, n_embedding), minval=-1, maxval=1,
                name='embedding'))
            self.embed = tf.nn.embedding_lookup(
                self.embedding, self.inputs, name='embed')

            # Create softmax layer
            softmax_w = tf.Variable(tf.truncated_normal(
                shape=(n_vocab, n_embedding), stddev=0.01))
            softmax_b = tf.Variable(tf.zeros(n_vocab))

            # Compute loss, cost and optimizer
            self.loss = tf.nn.sampled_softmax_loss(
                softmax_w, softmax_b, self.labels, self.embed, n_sampled,
                n_vocab)
            self.cost = tf.reduce_mean(self.loss)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

            # Create validation
            #
            # From Thushan Ganegedara's implementation
            # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id
            # implies more frequent
            self.valid_examples = np.array(random.sample(
                range(valid_window), valid_size // 2))
            self.valid_examples = np.append(self.valid_examples,
                                            random.sample(
                                                range(1000, 1000 +
                                                      valid_window),
                                                valid_size // 2))

            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # We use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(
                tf.square(self.embedding), 1, keep_dims=True))
            self.normalized_embedding = self.embedding / norm
            valid_embedding = tf.nn.embedding_lookup(
                self.normalized_embedding, valid_dataset)
            self.similarity = tf.matmul(valid_embedding,
                                        tf.transpose(
                                            self.normalized_embedding))

            # Define the saver
            self.saver = tf.train.Saver()

    def checkpoint_model(self, sess, path):
        self.saver(sess, path)


vocab_to_int, int_to_vocab, int_words = load_data()
train_words = subsampling(word_list=int_words, threshold=1e-5)

epochs = 10
batch_size = 1000
valid_size = 16
window_size = 10


vectorizer = WordVectorizer(int_to_vocab)

# Train the model
with tf.Session(graph=vectorizer.graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())
    log = 'Nearest to {}: {}'
    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        for x, y in batches:
            feed = {vectorizer.inputs: x,
                    vectorizer.labels: np.array(y)[:, None]}
            training_loss, _ = sess.run(
                [vectorizer.cost, vectorizer.optimizer], feed_dict=feed)
            loss += training_loss

            # Print training information
            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            # Perform quick validation
            if iteration % 1 == 0:
                # note that this is expensive (~20% slowdown if computed every
                # 500 steps)
                sim = sess.run(vectorizer.similarity)
                for i in range(valid_size):
                    valid_word = int_to_vocab[vectorizer.valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]

                    closest = ', '.join([int_to_vocab[nearest[k]]
                                         for k in range(top_k)])
                    print(log.format(valid_word, closest))

            iteration += 1
    # Save the final model
    vectorizer.checkpoint_model(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(vectorizer.normalized_embedding)


# Restore the latest model for examination
with tf.Session(graph=vectorizer.graph) as sess:
    vectorizer.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    embed_mat = sess.run(vectorizer.embedding)

# Plot the result
viz_words = 500
tsne = TSNE()
embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])

fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx],
                 (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
