import tensorflow as tf
from tensorflow.contrib import seq2seq
import numpy as np
from collections import Counter
import helper
import problem_unittests as tests

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token

    """
    punctuation_dict = {'.': '||period||',
                        ',': '||comma||',
                        '"': '||quotation_mark||',
                        ';': '||semicolon||',
                        '!': '||exclamation_mark||',
                        '?': '||question_mark||',
                        '(': '||left_parenthesis||',
                        ')': '||right_parenthesis||',
                        '--': '||dash||',
                        '\n': '||return||'}

    return punctuation_dict


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return input, targets, learning_rate


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm_layer = 1

    # Define the layer
    layer = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    # Stack the cell
    cell = tf.contrib.rnn.MultiRNNCell([layer] * lstm_layer)

    # Initialise the state
    initial_state = tf.identity(cell.zero_state(
        batch_size, tf.float32), name='initial_state')

    return cell, initial_state


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """

    embedding = tf.Variable(
        tf.random_uniform(shape=[vocab_size, embed_dim], minval=-1, maxval=1,
                          name='embedding'))
    embed = tf.nn.embedding_lookup(
        embedding, input_data, name='embed')
    return embed


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """

    outputs, state = tf.nn.dynamic_rnn(
        cell=cell, inputs=inputs, dtype=tf.float32)
    state = tf.identity(state, name='final_state')

    return outputs, state


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embed = get_embed(input_data, vocab_size, embed_dim)
    rnn_output, state = build_rnn(cell, embed)

    with tf.variable_scope('softmax'):
        logits = tf.contrib.layers.fully_connected(
            inputs=rnn_output, num_outputs=vocab_size, activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            biases_initializer=tf.zeros_initializer())
    return logits, state


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """

    word_length = len(int_text)
    batch_length = (batch_size * seq_length)
    n_batches = word_length // batch_length

    input_text = int_text[:n_batches * batch_length]

    target_text = input_text[1:] + [input_text[0]]

    batch = np.array(
        list(zip(np.split(np.array(input_text).reshape(batch_size, -1),
                          n_batches, 1),
                 np.split(np.array(target_text).reshape(batch_size, -1),
                          n_batches, 1))))
    return batch


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """

    input_tensor = loaded_graph.get_tensor_by_name('input:0')
    initial_state_tensor = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state_tensor = loaded_graph.get_tensor_by_name('final_state:0')
    probs_tensor = loaded_graph.get_tensor_by_name('probs:0')

    return input_tensor, initial_state_tensor, final_state_tensor, probs_tensor


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    next_word = np.random.choice(int_to_vocab.values(), p=probabilities)

    return next_word


# Unit tests
tests.test_create_lookup_tables(create_lookup_tables)
tests.test_tokenize(token_lookup)
tests.test_get_inputs(get_inputs)
tests.test_get_init_cell(get_init_cell)
tests.test_get_embed(get_embed)
tests.test_build_rnn(build_rnn)
tests.test_build_nn(build_nn)
tests.test_get_batches(get_batches)
tests.test_get_tensors(get_tensors)
tests.test_pick_word(pick_word)


# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

# Reload the data
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

# Number of Epochs
num_epochs = 50
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 250
# Embedding Dimension Size
embed_dim = 300
# Sequence Length
seq_length = 30
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 5

save_dir = './save'

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(
        cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var)
                        for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run(
                [cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


# Save parameters
helper.save_params((seq_length, save_dir))

# load parameters
_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()

# Generate the script
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word]
                      for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[dyn_seq_length - 1], int_to_vocab)

        gen_sentences.append(pred_word)

    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')

    print(tv_script)
