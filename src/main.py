import pathlib
import sys

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

_MAX_EPOCHS_NUM = 100
_STEPS_PER_EPOCH = 1024
_BATCH_SIZE = 128
_PREDICTION_BATCH_SIZE = 512
_VALIDATION_DATA_SIZE = 2048
_MIN_ACCURACY_IMPROVEMENT_DELTA = .1
_ACCURACY_IMPROVEMENT_PATIENCE = 3
_L2_REGULARIZATION_FACTOR = .01
_LEAKY_RELU_SLOPE = .1
_MAX_MINUTES_TIME_LIMIT = 55


def _get_model(max_given_cycle):
    """
    Get the binding prediction model.

    Args:
        max_given_cycle: The maximum cycle that we have data for.

    Returns:
         The prediction model.
    """
    model = keras.Sequential()

    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.LSTM(64))

    model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(_L2_REGULARIZATION_FACTOR)))
    model.add(layers.LeakyReLU(negative_slope=_LEAKY_RELU_SLOPE))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(5, activation='softmax'))

    model.add(AverageLast(5 - max_given_cycle))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='SparseCategoricalCrossentropy',
                  metrics=['accuracy'])

    return model


def _results_to_binding_predictions(model_results):
    # TODO
    return model_results[:, 4] + model_results[:, 3] - model_results[:, 0]


def _write_predictions(output_file_path, predictions):
    np.savetxt(output_file_path, predictions, fmt='%.4f')


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


def main():
    output_file_path = sys.argv[1]
    rna_compete_file_path = sys.argv[2]
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
              validation_data=val_ds,
              validation_steps=_VALIDATION_DATA_SIZE // _PREDICTION_BATCH_SIZE)

    # Evaluate the model on the RNAcompete dataset, and translate the model's outputs into binding
    # predictions.
    model_results = model.predict(rna_compete_ds)
    predictions = _results_to_binding_predictions(model_results)

    # Write the predictions to a file.
    _write_predictions(output_file_path, predictions)


if __name__ == "__main__":
    main()
