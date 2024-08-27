import pathlib
import pathlib
import pickle
import sys

import keras
import numpy as np
import tensorflow as tf
from keras.src.layers import Input, GRU, BatchNormalization, LeakyReLU, \
    Dense, Reshape, TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, ConvLSTM1D
from keras.src.optimizers import Adam

from rna_compete_dataset import rna_compete_dataset
from selex_dataset import combined_selex_dataset
from time_limit import TimeLimitCallback

_USE_EXPECTED_VALUE_FOR_PREDICTIONS = True
_SHOULD_SAVE_MODEL = False
_SAVED_MODEL_PATH = './saved_model.weights.h5'

_MAX_EPOCHS_NUM = 1000
_STEPS_PER_EPOCH = 2048
_BATCH_SIZE = 64
_PREDICTION_BATCH_SIZE = 512
_VALIDATION_DATA_SIZE = 32768
_L2_FACTOR = .01
_LEAKY_RELU_SLOPE = .1
_MAX_MINUTES_TIME_LIMIT = 55
_ADAM_LEARNING_RATE = 0.002


def _get_model1(last_dense_size):
    model = keras.Sequential()
    model.add(Input(shape=(None, 4)))

    model.add(Bidirectional(LSTM(16, return_sequences=True)))
    model.add(LSTM(16))
    model.add(Dropout(0.5))

    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.3))

    model.add(Dense(last_dense_size, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=_ADAM_LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def _get_model2(last_dense_size):
    model = keras.Sequential()
    model.add(Input(shape=(None, 4)))

    model.add(Bidirectional(GRU(16, return_sequences=True)))
    model.add(GRU(16))
    model.add(Dropout(0.3))

    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.3))

    model.add(Dense(last_dense_size, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=_ADAM_LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def _get_model3(last_dense_size):
    model = keras.Sequential()
    model.add(Input(shape=(None, 4)))

    model.add(Reshape((-1, 5, 4)))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(128)))

    model.add(GRU(64))
    model.add(Dropout(0.3))

    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(LeakyReLU(_LEAKY_RELU_SLOPE))

    model.add(Dense(last_dense_size, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=_ADAM_LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def _get_model4(last_dense_size):
    model = keras.Sequential()
    model.add(Input(shape=(None, 4)))

    model.add(Reshape((-1, 5, 4)))

    model.add(ConvLSTM1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(LeakyReLU(_LEAKY_RELU_SLOPE))

    model.add(Dense(last_dense_size, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=_ADAM_LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def _get_model5(last_dense_size):
    model = keras.Sequential()
    model.add(Input(shape=(None, 4)))

    model.add(Reshape((-1, 5, 4)))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(64)))

    model.add(GRU(32, return_sequences=True))
    model.add(GRU(16))

    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=_LEAKY_RELU_SLOPE))

    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=_LEAKY_RELU_SLOPE))

    model.add(Dense(last_dense_size, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=_ADAM_LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def _get_model(last_dense_size):
    """
    Get the binding prediction model.

    Returns:
         The prediction model.
    """
    return _get_model1(last_dense_size)


_TRAINING_CALLBACKS_LIST = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min', factor=0.5,
        patience=3,
        min_lr=1e-6),
    TimeLimitCallback(max_minutes=_MAX_MINUTES_TIME_LIMIT),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=.0001,
        patience=9,
        restore_best_weights=True)
]


def _results_to_binding_predictions(model_results, labels):
    if _USE_EXPECTED_VALUE_FOR_PREDICTIONS:
        return np.dot(model_results, labels)
    else:  # use argmax
        indices_of_max_prob = np.argmax(model_results, axis=1)
        return labels[indices_of_max_prob]


def _write_predictions(output_file_path, predictions):
    # Truncate the output file.
    with open(output_file_path, 'w') as _:
        pass
    # Write the predictions vector.
    np.savetxt(output_file_path, predictions, fmt='%.4f')

# TODO - remove when done:
# all the predictions are the same
# rewrite _results_to_binding_predictions
# bad correlation
# too good val_accuracy (too low val size?)
# write the report


def _parse_selex_file_paths(file_paths):
    """
    Parse the given SELEX file paths

    Args:
        file_paths: A list of the SELEX file paths, each one formatted as "RBP%d_%d.txt" where the
                    first integer is the protein index and the second integer is the cycle number.

    Returns:
        A dictionary that maps the cycle number to the corresponding file path.
    """
    ret = {}
    for path in file_paths:
        # The filename is formatted as "RBP%d_%d.txt".
        cycle = int(pathlib.Path(path).name.split('.')[0].split('_')[1])
        ret[cycle] = path
    return ret


def _get_selex_dataset(selex_file_paths, validation_data_size):
    """
    Get the SELEX dataset, split into training and validation sets.

    Args:
        selex_file_paths: A dictionary that maps the cycle number to the corresponding file path.
        validation_data_size: The size of the validation set.

    Returns:
         A pair of the training and validation sets.
    """
    train_ds, val_ds = combined_selex_dataset(selex_file_paths, validation_data_size)
    # train_ds = augment_dataset(train_ds)
    train_ds = train_ds.padded_batch(
        batch_size=_BATCH_SIZE,
        padded_shapes=([None, 4], []),
        padding_values=0)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE).padded_batch(
        batch_size=_PREDICTION_BATCH_SIZE,
        padded_shapes=([None, 4], []),
        padding_values=0)

    return train_ds, val_ds


def _get_tensors(dataset):
    """
    Transform a batched dataset into a pair of tensors - the first is the features and the second
    is the labels.

    Args:
        dataset: A Dataset instance.

    Returns:
        A pair of tensors - the first is the features and the second is the labels.
    """
    features, labels = [], []

    for features_batch, labels_batch in dataset:
        features.append(features_batch.numpy())
        labels.append(labels_batch.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def main():
    output_file_path = sys.argv[1]
    rna_compete_file_path = sys.argv[2]

    rna_compete_ds = rna_compete_dataset(rna_compete_file_path).padded_batch(
        batch_size=_PREDICTION_BATCH_SIZE, padded_shapes=[45, 4], padding_values=0)

    selex_file_paths = _parse_selex_file_paths(sys.argv[3:])

    train_ds, val_ds = _get_selex_dataset(selex_file_paths, _VALIDATION_DATA_SIZE)
    print(f'Creating a model with {len(selex_file_paths) + 1} classes')
    model = _get_model(len(selex_file_paths) + 1)
    model.summary()

    # Fit the model using the SELEX dataset, limiting the training time and stopping if the
    # validation accuracy stops improving.
    history = model.fit(train_ds,
                        epochs=_MAX_EPOCHS_NUM,
                        steps_per_epoch=_STEPS_PER_EPOCH,
                        callbacks=[_TRAINING_CALLBACKS_LIST],
                        validation_data=_get_tensors(val_ds),
                        validation_batch_size=_PREDICTION_BATCH_SIZE)

    if _SHOULD_SAVE_MODEL:
        model.save_weights(_SAVED_MODEL_PATH, overwrite=True)
        with open('./training_history1_39', 'wb') as f:
            pickle.dump(history.history, f)

    # Evaluate the model on the RNAcompete dataset, and translate the model's outputs into binding
    # predictions.
    model_results = model.predict(rna_compete_ds)
    # Translate model results to binding intensity predictions:
    labels = sorted(selex_file_paths.keys() | {0})
    predictions = _results_to_binding_predictions(model_results, labels)

    # Write the predictions to a file.
    _write_predictions(output_file_path, predictions)


if __name__ == "__main__":
    main()
