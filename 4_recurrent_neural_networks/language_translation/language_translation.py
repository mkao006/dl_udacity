import time
import numpy as np
import tensorflow as tf
import helper
import problem_unittests as tests


source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)


def inspect_data(source_text, target_text, view_sentence_range=(0, 10)):
    print('Dataset Stats')
    print('Roughly the number of unique words: {}'.format(
        len({word: None for word in source_text.split()})))

    sentences = source_text.split('\n')
    word_counts = [len(sentence.split()) for sentence in sentences]
    print('Number of sentences: {}'.format(len(sentences)))
    print('Average number of words in a sentence: {}'.format(
        np.average(word_counts)))

    print()
    print('English sentences {} to {}:'.format(*view_sentence_range))
    print('\n'.join(source_text.split('\n')[
        view_sentence_range[0]:view_sentence_range[1]]))
    print()
    print('French sentences {} to {}:'.format(*view_sentence_range))
    print('\n'.join(target_text.split('\n')[
        view_sentence_range[0]:view_sentence_range[1]]))


def text_to_ids(source_text, target_text, source_vocab_to_int,
                target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """

    source_id_text = [
        [source_vocab_to_int[word] for word in sentence.split()]
        for sentence in source_text.split('\n')]
    target_id_text = [
        [target_vocab_to_int[word] for word in sentence.split()] +
        [target_vocab_to_int['<EOS>']]
        for sentence in target_text.split('\n')]

    return source_id_text, target_id_text


# Inspect data
inspect_data(source_text, target_text, (0, 5))

# Unit tests
tests.test_text_to_ids(text_to_ids)

# Preprocess all the data and save it
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)


# Check point
((source_int_text, target_int_text),
 (source_vocab_to_int, target_vocab_to_int),
 _) = helper.load_preprocess()


def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    input = tf.placeholder(tf.int32, shape=[None, None], name='input')
    targets = tf.placeholder(tf.int32, shape=[None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input, targets, learning_rate, keep_prob


tests.test_model_inputs(model_inputs)


def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for decoding
    :param target_data: Target Placeholder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    go_id = target_vocab_to_int['<GO>']
    truncated_data = tf.strided_slice(
        input_=target_data,
        begin=[0, 0],
        end=[batch_size, -1],
        strides=[1, 1])
    start_signal = tf.fill(dims=[batch_size, 1], value=go_id)
    processed_decoding_input = tf.concat(
        [start_signal, truncated_data], axis=1)
    return processed_decoding_input


tests.test_process_decoding_input(process_decoding_input)


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    cell_with_dropout = tf.contrib.rnn.DropoutWrapper(
        cell=cell, output_keep_prob=keep_prob)
    encoder = tf.contrib.rnn.MultiRNNCell(
        cells=[cell_with_dropout] * num_layers)
    _, encoder_state = tf.nn.dynamic_rnn(cell=encoder,
                                         inputs=rnn_inputs,
                                         dtype=tf.float32)
    return encoder_state


tests.test_encoding_layer(encoding_layer)


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         sequence_length, decoding_scope, output_fn,
                         keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits

    """
    train_decoder_function = tf.contrib.seq2seq.simple_decoder_fn_train(
        encoder_state=encoder_state)
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        cell=dec_cell,
        decoder_fn=train_decoder_function,
        inputs=dec_embed_input,
        sequence_length=sequence_length,
        scope=decoding_scope)

    logit = output_fn(train_pred)

    return logit


tests.test_decoding_layer_train(decoding_layer_train)


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings,
                         start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope,
                         output_fn, keep_prob):
    # NOTE (Michael): Need to double check where the 'keep_prob' goes.
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: The maximum allowed time steps to decode
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
        output_fn=output_fn,
        encoder_state=encoder_state,
        embeddings=dec_embeddings,
        start_of_sequence_id=start_of_sequence_id,
        end_of_sequence_id=end_of_sequence_id,
        maximum_length=maximum_length - 1,
        num_decoder_symbols=vocab_size)
    infer_logit, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        cell=dec_cell,
        decoder_fn=infer_decoder_fn,
        scope=decoding_scope)

    return infer_logit


tests.test_decoding_layer_infer(decoding_layer_infer)


def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size,
                   sequence_length, rnn_size, num_layers, target_vocab_to_int,
                   keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size)
    cell_with_dropout = tf.contrib.rnn.DropoutWrapper(
        cell=cell, output_keep_prob=keep_prob)
    decoder = tf.contrib.rnn.MultiRNNCell(
        cells=[cell_with_dropout] * num_layers)

    with tf.variable_scope('decoding_scope') as decoding_scope:
        # NOTE (Michael): Need to double check the activation function
        output_fn = (lambda x: tf.contrib.layers.fully_connected(
            inputs=x,
            num_outputs=vocab_size,
            activation_fn=None,
            scope=decoding_scope))

        train_logit = decoding_layer_train(
            encoder_state=encoder_state,
            dec_cell=decoder,
            dec_embed_input=dec_embed_input,
            sequence_length=sequence_length,
            decoding_scope=decoding_scope,
            output_fn=output_fn,
            keep_prob=keep_prob)

    with tf.variable_scope('decoding_scope', reuse=True) as decoding_scope:
        start_of_sequence_id = target_vocab_to_int['<GO>']
        end_of_sequence_id = target_vocab_to_int['<EOS>']
        infer_logit = decoding_layer_infer(
            encoder_state=encoder_state,
            dec_cell=decoder,
            dec_embeddings=dec_embeddings,
            start_of_sequence_id=start_of_sequence_id,
            end_of_sequence_id=end_of_sequence_id,
            maximum_length=sequence_length - 1,
            vocab_size=vocab_size,
            decoding_scope=decoding_scope,
            output_fn=output_fn,
            keep_prob=keep_prob)

        return train_logit, infer_logit


tests.test_decoding_layer(decoding_layer)


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers,
                  target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """

    # Encode the source and output the state
    embedded_input = tf.contrib.layers.embed_sequence(
        ids=input_data,
        vocab_size=source_vocab_size,
        embed_dim=enc_embedding_size)

    encoder_state = encoding_layer(rnn_inputs=embedded_input,
                                   rnn_size=rnn_size,
                                   num_layers=num_layers,
                                   keep_prob=keep_prob)

    decoder_input = process_decoding_input(
        target_data=target_data,
        target_vocab_to_int=target_vocab_to_int,
        batch_size=batch_size)

    # Take in the state and processed target input and output the
    # training and inference logits
    decoder_embeddings_weights = tf.Variable(
        tf.random_uniform([target_vocab_size, dec_embedding_size]))

    decoder_embed_input = tf.nn.embedding_lookup(
        params=decoder_embeddings_weights,
        ids=decoder_input)

    # print('\ndec_embed_input is {}'.format(decoder_embed_input))
    # print('dec_embeddings is {}'.format(decoder_embeddings_weights))
    # print('encoder_state is {}'.format(encoder_state))
    # print('target_vocab_size is {}'.format(target_vocab_size))
    # print('sequence_length is {}'.format(sequence_length))
    # print('rnn_size is {}'.format(rnn_size))
    # print('num_layers is {}'.format(num_layers))
    # print('target_vocab_to_int is {}'.format(target_vocab_to_int))
    # print('keep_prob is {}'.format(keep_prob))

    train_logit, infer_logit = decoding_layer(
        dec_embed_input=decoder_embed_input,
        dec_embeddings=decoder_embeddings_weights,
        encoder_state=encoder_state,
        vocab_size=target_vocab_size,
        sequence_length=sequence_length,
        rnn_size=rnn_size,
        num_layers=num_layers,
        target_vocab_to_int=target_vocab_to_int,
        keep_prob=keep_prob)

    return train_logit, infer_logit


tests.test_seq2seq_model(seq2seq_model)


def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0, 0), (0, max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0, 0), (0, max_seq - logits.shape[1]), (0, 0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))


# Number of Epochs
epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 3
# Embedding Size
encoding_embedding_size = 256
decoding_embedding_size = 256
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.5

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = helper.pad_sentence_batch(source_int_text[:batch_size])
valid_target = helper.pad_sentence_batch(target_int_text[:batch_size])


save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int,
                                     target_vocab_to_int), _ = (
                                         helper.load_preprocess())

max_source_sentence_length = max([len(sentence)
                                  for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(
        max_source_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(
            source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var)
                            for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            start_time = time.time()

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})

            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})

            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(
                np.array(valid_target), batch_valid_logits)
            end_time = time.time()
            print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                  .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')

# Save parameters for checkpoint
helper.save_params(save_path)


# Check point
_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab,
                                                target_int_to_vocab) = helper.load_preprocess()

load_path = helper.load_params()


def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """

    sentence_ind = [vocab_to_int.get(word, vocab_to_int['<UNK>'])
                    for word in sentence.lower().split()]
    return sentence_ind


tests.test_sentence_to_seq(sentence_to_seq)


translate_sentence = 'he saw a old yellow truck .'
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(
        logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format(
    [source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format(
    [i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format(
    [target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))
