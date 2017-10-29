import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.constraints import maxnorm


def load_train_set():
    all_feature = list()
    all_label = list()
    for i in range(1, 6):
        feature, label = pickle.load(
            open('preprocess_batch_{}.p'.format(i), 'rb'))
        all_feature.extend(feature)
        all_label.extend(label)

    all_feature = np.array(all_feature)
    all_label = np.array(all_label)
    return all_feature, all_label


def load_validation_set():
    feature, label = pickle.load(open('preprocess_validation.p', 'rb'))
    return feature, label


def load_test_set():
    feature, label = pickle.load(open('preprocess_test.p', 'rb'))
    return feature, label


def cifar_model():
    # Model specification
    model = Sequential([
        Conv2D(input_shape=(32, 32, 3),
               filters=32, kernel_size=3, padding='same',
               kernel_constraint=maxnorm(3)),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=64, kernel_size=3, padding='same'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=64, kernel_size=3, padding='same'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(units=512, activation='relu',
              kernel_constraint=maxnorm(3)),
        Dropout(0.5),
        Dense(units=10, activation='softmax'),
    ])
    print(model.summary())

    # Compile the model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                 epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])

    return model


# Load the datasets
train_feature, train_label = load_train_set()
validation_feature, validation_label = load_validation_set()
test_feature, test_label = load_test_set()

# Specify and fit the model
model = cifar_model()
model.fit(train_feature, train_label, epochs=10, batch_size=450,
          validation_data=(validation_feature, validation_label))

# Test the model
loss_and_metrics = model.evaluate(
    test_feature, test_label, batch_size=100)
print('Test accuracy: {:.6f} \n'.format(loss_and_metrics[1]))
