import pathlib
import sys
import os

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras import optimizers
from keras import regularizers

from average_last_cycles import AverageLast
from dna_dataset_utils import augment_dataset
from rna_compete_dataset import rna_compete_dataset
from selex_dataset import combined_selex_dataset
from time_limit import TimeLimitCallback

SAVED_MODEL_PATH = 'saved_model'

_MAX_EPOCHS_NUM = 1000
_STEPS_PER_EPOCH = 2048
_BATCH_SIZE = 128
_PREDICTION_BATCH_SIZE = 512
_VALIDATION_DATA_SIZE = 32768
_MIN_ACCURACY_IMPROVEMENT_DELTA = .0001
_ACCURACY_IMPROVEMENT_PATIENCE = 5
_L2_REGULARIZATION_FACTOR = .01
_LEAKY_RELU_SLOPE = .1
_MAX_MINUTES_TIME_LIMIT = 55
_ADAM_LEARNING_RATE = 0.001


def _get_model(max_given_cycle):
    """
    Get the binding prediction model.

    Args:
        max_given_cycle: The maximum cycle that we have data for.

    Returns:
         The prediction model.
    """
    model = keras.Sequential()

    model.add(layers.LSTM(1024))

    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(_L2_REGULARIZATION_FACTOR)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=_LEAKY_RELU_SLOPE))

    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(_L2_REGULARIZATION_FACTOR)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=_LEAKY_RELU_SLOPE))

    model.add(layers.Dense(5, activation='softmax'))

    model.add(AverageLast(5 - max_given_cycle))

    model.compile(optimizer=optimizers.Adam(learning_rate=_ADAM_LEARNING_RATE),
                  loss='SparseCategoricalCrossentropy',
                  metrics=['accuracy'])

    return model


def _get_model2(max_given_cycle, vocab_size, embedding_dim, max_sequence_length):
    # Not used. Another option for the net structure.
    model = keras.Sequential()

    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))

    model.add(layers.Bidirectional(layers.LSTM(512, return_sequences=True, recurrent_dropout=0.3)))
    model.add(layers.Bidirectional(layers.LSTM(512)))

    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(_L2_REGULARIZATION_FACTOR)))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.LeakyReLU(negative_slope=_LEAKY_RELU_SLOPE))

    model.add(layers.Dense(5, activation='softmax'))

    model.add(AverageLast(5 - max_given_cycle))

    model.compile(optimizer=optimizers.Adam(learning_rate=_ADAM_LEARNING_RATE),
                  loss='SparseCategoricalCrossentropy',
                  metrics=['accuracy'])

    return model


def _get_model3(vocab_size, max_sequence_length, num_classes, embedding_dim=16):
    # Not used. Another option for the net structure.

    model = keras.Sequential()

    # Embedding layer for nucleotide sequences
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))

    # Convolutional layers to extract local patterns/motifs
    model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Bi-directional LSTM layer to capture sequence dependencies
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))

    # Attention layer for focusing on important sequence parts
    model.add(layers.Attention())

    # Global pooling to reduce dimensions
    model.add(layers.GlobalMaxPooling1D())

    # Dense layers for complex patterns
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.3))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(rate=0.3))

    model.add(layers.Dense(num_classes, activation='softmax'))
    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=loss,
                  metrics=metrics)

    return model


def _results_to_binding_predictions(model_results):
    # TODO
    return model_results[:, 4] + model_results[:, 3] - model_results[:, 0]


def _write_predictions(output_file_path, predictions):
    np.savetxt(output_file_path, predictions, fmt='%.4f')


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
    train_ds = augment_dataset(train_ds).padded_batch(
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


def _use_saved_model_for_rna_compete_ds(output_file_path, rna_compete_file_path):
    model = keras.models.load_model(SAVED_MODEL_PATH)
    # Evaluate the model on the RNAcompete dataset, and translate the model's outputs into binding
    # predictions.
    rna_compete_ds = rna_compete_dataset(rna_compete_file_path).padded_batch(
        batch_size=_PREDICTION_BATCH_SIZE, padded_shapes=[None, 4], padding_values=0)
    model_results = model.predict(rna_compete_ds)
    predictions = _results_to_binding_predictions(model_results)

    # Write the predictions to a file.
    _write_predictions(output_file_path, predictions)


def main():
    output_file_path = sys.argv[1]
    rna_compete_file_path = sys.argv[2]

    # use saved model to adjust _results_to_binding_predictions:
    if os.path.exists(SAVED_MODEL_PATH):
        print("Loading saved model...")
        _use_saved_model_for_rna_compete_ds(output_file_path, rna_compete_file_path)
        return

    # if no saved model, follow original workflow:
    selex_file_paths = _parse_selex_file_paths(sys.argv[3:])
    max_given_cycle = max(selex_file_paths.keys())

    train_ds, val_ds = _get_selex_dataset(selex_file_paths, _VALIDATION_DATA_SIZE)
    rna_compete_ds = rna_compete_dataset(rna_compete_file_path).padded_batch(
        batch_size=_PREDICTION_BATCH_SIZE, padded_shapes=[None, 4], padding_values=0)
    model = _get_model(max_given_cycle)

    time_limit_callback = TimeLimitCallback(max_minutes=_MAX_MINUTES_TIME_LIMIT)
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        min_delta=_MIN_ACCURACY_IMPROVEMENT_DELTA,
        patience=_ACCURACY_IMPROVEMENT_PATIENCE,
        restore_best_weights=True)

    # Fit the model using the SELEX dataset, limiting the training time and stopping if the
    # validation accuracy stops improving.
    model.fit(train_ds,
              epochs=_MAX_EPOCHS_NUM,
              steps_per_epoch=_STEPS_PER_EPOCH,
              callbacks=[time_limit_callback, early_stopping_callback],
              validation_data=_get_tensors(val_ds),
              validation_batch_size=_PREDICTION_BATCH_SIZE)

    #  by default, save a model after it was trained
    model.save(SAVED_MODEL_PATH)

    # Evaluate the model on the RNAcompete dataset, and translate the model's outputs into binding
    # predictions.
    model_results = model.predict(rna_compete_ds)
    predictions = _results_to_binding_predictions(model_results)

    # Write the predictions to a file.
    _write_predictions(output_file_path, predictions)


if __name__ == "__main__":
    main()
