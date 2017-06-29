import numpy as np
import pandas as pd

# Load the data
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)

# Create dummy variables
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)

# normalise continuous variables
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean) / std


# Split into training and test
# Save data for approximately the last 21 days
test_data = data[-21 * 24:]

# Now remove the test data from the data set
data = data[:-21 * 24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(
    target_fields, axis=1), test_data[target_fields]

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # Replace 0 with your sigmoid calculation.
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

        # If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your
        # implementation there instead.
        #
        # def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid

    def train(self, features, targets):
        '''Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            #### Implement the forward pass here ####
            ### Forward pass ###
            # TODO: Hidden layer - Replace these values with your calculations.

            # signals into hidden layer
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            # signals from hidden layer
            hidden_outputs = self.activation_function(hidden_inputs)

            # TODO: Output layer - Replace these values with your calculations.
            # signals into final output layer
            final_inputs = np.dot(hidden_outputs,
                                  self.weights_hidden_to_output)
            # signals from final output layer
            final_outputs = final_inputs

            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error - Replace this value with your calculations.
            # Output layer error is the difference between desired target and
            # actual output.
            error = y - final_outputs

            # TODO: Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(error, self.weights_hidden_to_output.T)

            # TODO: Backpropagated error terms - Replace these values with your
            # calculations.
            output_error_term = error
            hidden_derivative = hidden_outputs * (1 - hidden_outputs)
            hidden_error_term = hidden_error * hidden_derivative

            # print(hidden_error_term.shape)
            # print(X.shape)
            # print(output_error_term.shape)
            # print(hidden_outputs.shape)

            # Weight step (input to hidden)
            delta_weights_i_h += X[:, None] * hidden_error_term
            # Weight step (hidden to output)
            delta_weights_h_o += hidden_outputs[:, None] * output_error_term

        # TODO: Update the weights - Replace these values with your
        # calculations.
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        '''Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values

        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the
        # appropriate calculations.
        # signals into hidden layer
        hidden_inputs = np.dot(features,
                               self.weights_input_to_hidden)
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: Output layer - Replace these values with the appropriate
        # calculations.
        # signals into final output layer
        final_inputs = np.dot(hidden_outputs,
                              self.weights_hidden_to_output)
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs


def MSE(y, Y):
    return np.mean((y - Y)**2)


nn = NeuralNetwork(input_nodes=56, hidden_nodes=4,
                   output_nodes=1, learning_rate=0.005)
test = nn.run(features=train_features)
test = nn.train(features=train_features, targets=train_targets['cnt'])


inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])
network = NeuralNetwork(3, 2, 1, 0.5)
network.weights_input_to_hidden = test_w_i_h.copy()
network.weights_hidden_to_output = test_w_h_o.copy()
network.run(inputs)


import sys

### Set the hyperparameters here ###
iterations = 1000
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train': [], 'validation': []}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

    network.train(X, y)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T,
                     train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii / float(iterations))
                     + "% ... Training loss: " + str(train_loss)[:5]
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
