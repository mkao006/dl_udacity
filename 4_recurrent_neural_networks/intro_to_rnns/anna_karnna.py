import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''

    # Calculate the number of batches
    character_per_batch = n_seqs * n_steps
    n_batches = len(arr) // character_per_batch

    # Discard the extra text
    arr = arr[:n_batches * character_per_batch]

    # Reshape the array
    arr = arr.reshape((n_seqs, -1))

    # Create the generator function
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n + n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


def build_inputs(batch_size, num_steps):
    '''Define placeholders for inputs, targets, and dropout

        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch

    '''

    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(
        tf.int32, [batch_size, num_steps], name='targets')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.

        Arguments
        ---------
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size

    '''

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state


def build_output(lstm_output, in_size, out_size):
    ''' Build a softmax layer, return the softmax output and logits.

        Arguments
        ---------

        x: Input tensor
        in_size: Size of the input tensor, for example, size of the LSTM cells
        out_size: Size of this softmax layer

    '''

    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal(
            [in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.constant(0.1, shape=[out_size]))

    logits = tf.add(tf.matmul(x, softmax_w), softmax_b)
    out = tf.nn.softmax(logits, name='predictions')

    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    ''' Calculate the loss from the logits and the targets.

        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        lstm_size: Number of LSTM hidden units
        num_classes: Number of classes in targets

    '''

    y_onehot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_onehot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y_reshaped)
    summed_loss = tf.reduce_mean(loss)

    return summed_loss


def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.

        Arguments:
        loss: Network loss
        learning_rate: Learning rate for optimizer
        grad_clip: The value the gradient should be clipped to avoid explosion

    '''

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimiser = train_op.apply_gradients(zip(grads, tvars))

    return optimiser


class CharRNN:
    ''' Initialise a character RNN.

        Arguments:
        num_classes: Number of classes in targets
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        learning_rate: Learning rate for optimizer
        grad_clip: The value the gradient should be clipped to avoid explosion
        sampling: Whether the model is used to generate new text.

    '''

    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):

        if sampling:
            batch_size, num_steps = 1, 1

        tf.reset_default_graph()

        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(
            batch_size, num_steps)

        # Build the LSTM cell
        cell, self.initial_state = build_lstm(
            lstm_size, num_layers, batch_size, self.keep_prob)

        # Run the data through the RNNlayers
        x_onehot = tf.one_hot(self.inputs, num_classes)

        # Run each sequence step through the RNN and collect the outputs
        outputs, state = tf.nn.dynamic_rnn(
            cell, x_onehot, initial_state=self.initial_state)
        self.final_state = state

        # Get softmax predictions and logits
        self.prediction, self.logits = build_output(
            outputs, lstm_size, num_classes)

        # Loss and optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets,
                               lstm_size, num_classes)
        self.optimiser = build_optimizer(self.loss, learning_rate, grad_clip)


def train_charrnn(data,
                  num_classes,
                  batch_size=100,
                  num_steps=100,
                  lstm_size=512,
                  num_layers=2,
                  learning_rate=0.001,
                  save_every_n=100,
                  keep_prob=0.5,
                  epochs=20):
    ''' A wrapper to train and save the model.

        Arguments:
        data: The numerically encoded array of the text.
        num_classes: Number of classes in targets
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        learning_rate: Learning rate for optimizer
        grad_clip: The value the gradient should be clipped to avoid explosion
        sampling: Whether the model is used to generate new text.
        keep_prob: The dropout keep probability
        epochs: The number of epochs for training.

    '''

    model = CharRNN(num_classes=num_classes,
                    batch_size=batch_size,
                    num_steps=num_steps,
                    lstm_size=lstm_size,
                    num_layers=num_layers,
                    learning_rate=learning_rate)

    with tf.Session() as sess:
        # Initialise global variables
        tf.global_variables_initializer().run()

        # Initialise LSTM state
        state = sess.run(model.initial_state)
        loss = list()
        counter = 1

        # Create the saver, None means keep all of the check points
        saver = tf.train.Saver(max_to_keep=None)
        # Start the training
        for ep in range(epochs):
            # Load the data in batch
            for x, y in get_batches(data, batch_size, num_steps):
                start = time.time()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: state}
                # Train the model
                batch_loss, state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimiser],
                                                feed_dict=feed)
                end = time.time()
                loss.append(batch_loss)

                # Print training info
                print('Epoch {}/{} ...'.format(ep + 1, epochs),
                      'Training Step: {} ...'.format(counter),
                      'Training Loss: {} ...'.format(batch_loss),
                      '{:.4f} sec/batch'.format((end - start))
                      )
                counter += 1

                # Checkpoint the model
                if (counter % save_every_n == 0):
                    saver.save(sess,
                               'checkpoints/i{}_l{}.ckpt'.format(counter, lstm_size))

        # Save the final model
        saver.save(sess, 'checkpoints/i{}_l{}.ckpt'.format(counter, lstm_size))
    return loss


def pick_top_n(preds, vocab_size, top_n=5):
    '''This function picks the top n character, the conditional
    distribution comes from the RNN.

        Arguments:
        preds: The prediction from the RNN model.
        vocab_size: Number of classes in targets, should be same as 'num_classes'.
        top_n: The The top vocabs to be used.

    '''

    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def generator(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    ''' Function to generate new text based on the RNN model.

        Arguments:
        checkpoint: The loaded model checkpoint.
        n_samples: The size of the output sample.
        lstm_size: Size of the hidden layers in the LSTM cells.
        vocab_size: Number of classes in targets, should be same as 'num_classes'.
        prime: The starting characters of the new text.
    '''

    samples = [c for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

    return ''.join(samples)



# Load and process the data
with open('anna.txt', 'r') as f:
    text = f.read()
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


# Parameters
batch_size = 100        # Sequences per batch
num_steps = 100         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability
epochs = 20             # Number of epcohs
save_every_n = 200      # Save the model every k steps. (200 == each batch)


# Train the model
train_loss = train_charrnn(encoded,
                           num_classes=len(vocab),
                           batch_size=batch_size,
                           num_steps=num_steps,
                           lstm_size=lstm_size,
                           num_layers=num_layers,
                           learning_rate=learning_rate,
                           save_every_n=save_every_n,
                           keep_prob=keep_prob,
                           epochs=epochs)


# Generate new text. Various check point can be loaded to examine the training.
checkpoint = tf.train.latest_checkpoint('checkpoints')
# checkpoint = 'checkpoints/i200_l512.ckpt'
samp = generator(checkpoint, 2000, lstm_size, len(vocab), prime="Far")
print(samp)

# Plot the training loss
plt.plot(train_loss)
plt.ylabel('Training Loss')
plt.xlabel('Training Iterations')
plt.show()
