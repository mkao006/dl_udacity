# Steps for training a seq2seq model.
#
# Data processing:
#     - Create dictionary to convert words in to index.
#     - Append special start and end tokens.
#     - Pad sequence to maximum length. (7 in this example)
#     - Pad target sequence with start token
#
# Model:
#     - Create embedding layer for input sequence.
#     - Create LSTM to generate hidden state
#     - Create embedding layer for target sequence.
#     - Create LSTM to decode both the target sequence and the hidden state
#       from encoder.
#     - Create fullly connected layer for decoder LSTM output.
#     - Create trainer for the decoder LSTM
#     - Create inference decoder, this actually generates the prediction.
#     - Train the model


import tensorflow as tf
import numpy as np
import helper

source_path = 'data/letters_source.txt'
target_path = 'data/letters_target.txt'

source_sentences = helper.load_data(source_path).split()
target_sentences = helper.load_data(target_path).split()

# params
start_token = '<s>'
end_token = '<\s>'
unknown_token = '<unk>'
pad_token = '<pad>'
epochs = 1
batch_size = 128
rnn_size = 50
num_layers = 2
encoding_embedding_size = 13
decoding_embedding_size = 13
learning_rate = 0.001


class SeqToSeq:

    def __init__(self,
                 source,
                 target,
                 start_token='<s>',
                 end_token='<\s>',
                 unknown_token='<unk>',
                 pad_token='<pad>',
                 epochs=60,
                 batch_size=128,
                 rnn_size=50,
                 num_layers=2,
                 encoding_embedding_size=13,
                 decoding_embedding_size=13,
                 learning_rate=0.001):
        ''' Defines the processing and model hyperparameters.
        '''
        self.start_token = start_token
        self.end_token = end_token
        self.unknown_token = unknown_token
        self.pad_token = pad_token
        self.epochs = epochs
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.encoding_embedding_size = encoding_embedding_size
        self.decoding_embedding_size = decoding_embedding_size
        self.learning_rate = learning_rate
        self.special_tokens = [self.start_token,
                               self.end_token,
                               self.unknown_token,
                               self.pad_token]

        self.max_seq_len = max([len(item) for item in source + target])

        self.source_ind, self.target_ind = (
            self._convert_sequence_to_ind(source, target))

        self.train_source = self.source_ind[self.batch_size:]
        self.train_target = self.target_ind[self.batch_size:]

        self.valid_source = self.source_ind[:self.batch_size]
        self.valid_target = self.target_ind[:self.batch_size]

    def _batch_generator(self, source, target):
        self.n_batches = len(source) // self.batch_size
        truncated_sample_size = self.n_batches * self.batch_size
        truncated_source = source[:truncated_sample_size]
        truncated_target = target[:truncated_sample_size]

        for start in range(0, truncated_sample_size, self.batch_size):
            end = start + self.batch_size
            yield truncated_source[start:end], truncated_target[start:end]

    def _convert_sequence_to_ind(self, source, target, padding=True):
        '''Function to convert the source and target to indexes using the
        dictionary constructed. Sequences are also padded.

        '''
        complete_sequence = source + target

        set_words = set([character
                         for item in complete_sequence
                         for character in item])
        complete_set = self.special_tokens + list(set_words)
        self.int_to_vocab = {word_ind: word
                             for word_ind, word in enumerate(complete_set)}
        self.vocab_to_int = {word: word_ind
                             for word_ind, word in self.int_to_vocab.items()}
        self.vocab_size = len(self.vocab_to_int)
        unknown_ind = self.vocab_to_int[self.unknown_token]
        source_sequence_ind = [
            [self.vocab_to_int.get(letter, unknown_ind) for letter in item]
            for item in source]
        target_sequence_ind = [
            [self.vocab_to_int.get(letter, unknown_ind) for letter in item]
            for item in target]

        if padding:
            padding_ind = [self.vocab_to_int[self.unknown_token]]
            source_sequence_ind = [seq +
                                   padding_ind * (self.max_seq_len - len(seq))
                                   for seq in source_sequence_ind]
            target_sequence_ind = [seq +
                                   padding_ind * (self.max_seq_len - len(seq))
                                   for seq in target_sequence_ind]

        return source_sequence_ind, target_sequence_ind

    def initialise_graph(self):
        # The graph should be defined here.
        self.graph = tf.Graph()
        with tf.Session(graph=self.graph):
            # Define placeholders
            self.source = tf.placeholder(tf.int32,
                                         shape=[self.batch_size,
                                                self.max_seq_len],
                                         name='source')
            self.target = tf.placeholder(tf.int32,
                                         shape=[self.batch_size,
                                                self.max_seq_len],
                                         name='target')

            # Define encoding embedding
            #
            # NOTE (Michael): Do we need the vocab size? TO me it make
            #                 sense that it is required. If this is
            #                 the case, then we may need to move the
            #                 creation of graph after the data has
            #                 been processed.
            self.encoder_embed = (
                tf.contrib.layers.embed_sequence(
                    ids=self.source,
                    vocab_size=self.vocab_size,
                    embed_dim=self.encoding_embedding_size))

            # Define encoding LSTM
            #
            # NOTE (Michael): Can we implement dropout here?
            encoder_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=self.rnn_size)
            encoder = tf.contrib.rnn.MultiRNNCell(
                cells=[encoder_cell] * self.num_layers)

            # NOTE (Michael): We don't need the output of the RNN,
            #                 since only the state is passed to the
            #                 decoder.
            _, self.encoder_state = tf.nn.dynamic_rnn(cell=encoder,
                                                      inputs=self.encoder_embed,
                                                      dtype=tf.float32)

            # Define decoder input
            #
            # NOTE (Michael): This implies that we need to move the
            #                 construction of the grph after the data
            #                 processing since we don't have the
            #                 'vocab_to_int' dictionary yet!!
            start_ind = self.vocab_to_int[self.start_token]
            self.decoder_input = tf.concat(
                [tf.fill([batch_size, 1], start_ind),
                 tf.strided_slice(input_=self.target,
                                  begin=[0, 0],
                                  end=[self.batch_size, -1],
                                  strides=[1, 1])],
                axis=1
            )

            # Define decoder embedding
            self.decoder_embed_weights = (
                tf.Variable(tf.random_uniform([self.vocab_size,
                                               self.decoding_embedding_size]),
                            name='decoder_embed_weights'))
            self.decoder_embed = tf.nn.embedding_lookup(
                params=self.decoder_embed_weights,
                ids=self.decoder_input)

            # Define decoder LSTM
            decoder_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=self.rnn_size)
            decoder = tf.contrib.rnn.MultiRNNCell(
                cells=[decoder_cell] * self.num_layers)

            # Decode the output of LSTM and generate prediction
            with tf.variable_scope('decoding') as decoding_scope:
                # Output Layer
                output_fn = (
                    lambda x: tf.contrib.layers.fully_connected(
                        inputs=x,
                        num_outputs=self.vocab_size,
                        activation_fn=None,
                        scope=decoding_scope))
                # Training Decoder
                train_decoder_fn = (
                    tf.contrib.seq2seq.simple_decoder_fn_train(
                        encoder_state=self.encoder_state))
                train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                    cell=decoder,
                    decoder_fn=train_decoder_fn,
                    inputs=self.decoder_embed,
                    sequence_length=self.max_seq_len,
                    scope=decoding_scope)
                train_logits = output_fn(train_pred)

            with tf.variable_scope('decoding', reuse=True) as decoding_scope:
                # Inference Decoder
                infer_decoder_fn = (
                    tf.contrib.seq2seq.simple_decoder_fn_inference(
                        output_fn=output_fn,
                        encoder_state=self.encoder_state,
                        embeddings=self.decoder_embed_weights,
                        start_of_sequence_id=self.vocab_to_int[self.start_token],
                        end_of_sequence_id=self.vocab_to_int[self.end_token],
                        maximum_length=self.max_seq_len - 1,
                        num_decoder_symbols=self.vocab_size))
                self.inference_logits, _, _ = (
                    tf.contrib.seq2seq.dynamic_rnn_decoder(cell=decoder,
                                                           decoder_fn=infer_decoder_fn,
                                                           scope=decoding_scope))

            # Define the loss function and optimiser
            self.loss = (
                tf.contrib.seq2seq.sequence_loss(logits=train_logits,
                                                 targets=self.target,
                                                 weights=tf.ones([self.batch_size,
                                                                  self.max_seq_len])))
            self.optimiser = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)

            # Gradient Clipping
            gradients = self.optimiser.compute_gradients(self.loss)
            capped_gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var)
                                for grad, var in gradients if grad is not None]
            self.train_ops = self.optimiser.apply_gradients(capped_gradients)

    def train(self):
        ''' Method to train the seq2seq RNN.
        '''

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                for batch, (source_batch, target_batch) in enumerate(
                        self._batch_generator(self.train_source,
                                              self.train_target)):
                    _, loss = sess.run([self.train_ops, self.loss],
                                       feed_dict={
                                           self.source: source_batch,
                                           self.target: target_batch})

                    batch_train_logits = sess.run(
                        self.inference_logits,
                        feed_dict={self.source: source_batch})
                    batch_valid_logits = sess.run(
                        self.inference_logits,
                        feed_dict={self.source: self.valid_source})

                    train_acc = np.mean(
                        np.equal(target_batch,
                                 np.argmax(batch_train_logits, axis=2)))
                    valid_acc = np.mean(
                        np.equal(self.valid_target,
                                 np.argmax(batch_valid_logits, axis=2)))
                    print('''Epoch {:>3} Batch {:>4}/{} Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'''
                          .format(epoch,
                                  batch,
                                  self.n_batches,
                                  train_acc,
                                  valid_acc,
                                  loss))

    def respond(self, input_sentence):
        ''' Method to give a response based on new input.
        '''

        unknown_ind = self.vocab_to_int[self.unknown_token]

        input_sentence_ind = [self.vocab_to_int.get(char, unknown_ind)
                              for char in input_sentence]

        padding_ind = [self.vocab_to_int[self.unknown_token]]
        input_sentence_ind = input_sentence_ind + \
            padding_ind * (self.max_seq_len - len(input_sentence_ind))

        batch_shell = np.zeros((self.batch_size, self.max_seq_len))
        batch_shell[0] = input_sentence_ind
        with tf.Session(graph=self.graph) as sess:
            chatbot_logits = sess.run(
                self.inference_logits, {self.source: batch_shell})[0]

            print('Input')
            print('  Word Ids:      {}'.format(
                [i for i in input_sentence_ind]))
            print('  Input Words: {}'.format(
                [self.int_to_vocab[i] for i in input_sentence_ind]))

            print('\nPrediction')
            print('  Word Ids:      {}'.format(
                [i for i in np.argmax(chatbot_logits, 1)]))
            print('  Chatbot Answer Words: {}'.format(
                [self.int_to_vocab[i] for i in np.argmax(chatbot_logits, 1)]))


model = SeqToSeq(source=source_sentences,
                 target=target_sentences,
                 start_token=start_token,
                 end_token=end_token,
                 unknown_token=unknown_token,
                 pad_token=pad_token,
                 epochs=epochs,
                 batch_size=batch_size,
                 rnn_size=rnn_size,
                 num_layers=num_layers,
                 encoding_embedding_size=encoding_embedding_size,
                 decoding_embedding_size=decoding_embedding_size,
                 learning_rate=learning_rate)
model.initialise_graph()
model.train()
model.respond('hello')
