import pathlib
import sys

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras import optimizers
from keras import regularizers

from dna_dataset_utils import augment_dataset
from rna_compete_dataset import rna_compete_dataset
from selex_dataset import combined_selex_dataset
from time_limit import TimeLimitCallback

_MAX_EPOCHS_NUM = 100
_STEPS_PER_EPOCH = 25000
_BATCH_SIZE = 128
_PREDICTION_BATCH_SIZE = 512
_VALIDATION_DATA_SIZE = 2000
_MIN_ACCURACY_IMPROVEMENT_DELTA = .1
_ACCURACY_IMPROVEMENT_PATIENCE = 3
_L2_REGULARIZATION_FACTOR = .01
_LEAKY_RELU_SLOPE = .1
_MAX_MINUTES_TIME_LIMIT = 55


def _get_model():
    """
    Get the binding prediction model.

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


def main():
    output_file_path = sys.argv[1]
    rna_compete_file_path = sys.argv[2]
    selex_file_paths = _parse_selex_file_paths(sys.argv[3:])

    selex_ds = (augment_dataset(combined_selex_dataset(selex_file_paths))
                .padded_batch(batch_size=_BATCH_SIZE, padded_shapes=([None, 4], []),
                              padding_values=0))
    rna_compete_ds = rna_compete_dataset(rna_compete_file_path).padded_batch(
        batch_size=_PREDICTION_BATCH_SIZE, padded_shapes=[None, 4], padding_values=0)
    model = _get_model()

    # Split the dataset into training and validation sets.
    val_ds = selex_ds.take(_VALIDATION_DATA_SIZE)
    train_ds = selex_ds.skip(_VALIDATION_DATA_SIZE).prefetch(tf.data.AUTOTUNE)

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
              validation_data=val_ds)

    # Evaluate the model on the RNAcompete dataset, and translate the model's outputs into binding
    # predictions.
    model_results = model.predict(rna_compete_ds)
    predictions = _results_to_binding_predictions(model_results)

    # Write the predictions to a file.
    _write_predictions(output_file_path, predictions)


if __name__ == "__main__":
    main()
