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

SAVED_MODEL_PATH = './RNA-Binding-Predictor/saved_model.keras'

_MAX_EPOCHS_NUM = 1000
_STEPS_PER_EPOCH = 2048
_BATCH_SIZE = 64
_PREDICTION_BATCH_SIZE = 512
_VALIDATION_DATA_SIZE = 32768
_MIN_ACCURACY_IMPROVEMENT_DELTA = .0001
_ACCURACY_IMPROVEMENT_PATIENCE = 5
_L2_REGULARIZATION_FACTOR = .01
_LEAKY_RELU_SLOPE = .1
_MAX_MINUTES_TIME_LIMIT = 55
_ADAM_LEARNING_RATE = 0.002

MAX_SEQUENCE_LENGTH = 40
NUM_CLASSES = 4
FILTERS = 64
KERNEL_SIZE = 5
POOL_SIZE = 2
DENSE_UNITS_1 = 128
DENSE_UNITS_2 = 64
NUM_HEADS = 4
KEY_DIM = 64


def _get_model():
    """
    Get the binding prediction model.

    Args:
        max_given_cycle: The maximum cycle that we have data for.

    Returns:
         The prediction model.
    """
    model = keras.Sequential()

    # LSTM layers with dropout and regularization
    model.add(layers.LSTM(256, return_sequences=True, input_shape=(40, 4),
                          kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.LSTM(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))

    # Dense layers with Batch Normalization and LeakyReLU activation
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))

    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.01))

    # Output layer for classification
    model.add(layers.Dense(5, activation='softmax'))

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=_ADAM_LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath=SAVED_MODEL_PATH, monitor='val_loss', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]


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
    train_ds = train_ds.padded_batch(  # augment_dataset(train_ds).padded_batch(
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
    model = _get_model()

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
              callbacks=[callbacks_list],
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
