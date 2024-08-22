import sys

import keras
from keras import layers
from keras import optimizers
from keras import regularizers

from dna_dataset_utils import augment_dataset
from rna_compete_dataset import rna_compete_dataset
from selex_dataset import combined_selex_dataset
from time_limit import TimeLimitCallback

_BATCH_SIZE = 32
_MAX_EPOCHS_NUM = 5000
_VALIDATION_DATA_SIZE = 1000
_MIN_DELTA = .1
_PATIENCE = 50
_L2_REGULARIZATION = .01
_LEAKY_RELU_SLOPE = .1
_MAX_MINUTES_TIME_LIMIT = 60


def _get_model():
    model = keras.Sequential()

    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(64))

    model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(_L2_REGULARIZATION)))
    model.add(layers.LeakyReLU(negative_slope=_LEAKY_RELU_SLOPE))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(4, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='SparseCategoricalCrossentropy',
                  metrics=['accuracy'])

    return model


def _results_to_binding_predictions(model_results):
    pass  # TODO


def _write_predictions(output_file_path, predictions):
    pass  # TODO


def _parse_file_paths(file_paths):
    ret = {}
    for path in file_paths:
        # The file path is formatted as "RBP%d_%d.txt".
        cycle = int(path.split('.')[0].split('_')[1])
        ret[cycle] = path
    return ret


def main():
    output_file_path = sys.argv[1]
    rna_compete_file_path = sys.argv[2]
    selex_file_paths = _parse_file_paths(sys.argv[3:])

    selex_ds = augment_dataset(combined_selex_dataset(selex_file_paths)).batch(_BATCH_SIZE)
    model = _get_model()
    time_limit_callback = TimeLimitCallback(max_minutes=_MAX_MINUTES_TIME_LIMIT)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                            min_delta=_MIN_DELTA,
                                                            patience=_PATIENCE,
                                                            restore_best_weights=True)

    # Manually split the dataset into training and validation sets.
    val_ds = selex_ds.take(_VALIDATION_DATA_SIZE)
    train_ds = selex_ds.skip(_VALIDATION_DATA_SIZE)
    model.fit(train_ds,
              epochs=_MAX_EPOCHS_NUM,
              callbacks=[time_limit_callback, early_stopping_callback],
              validation_data=val_ds)

    rna_compete_ds = rna_compete_dataset(rna_compete_file_path)
    model_results = model.predict(rna_compete_ds)
    predictions = _results_to_binding_predictions(model_results)

    _write_predictions(output_file_path, predictions)


if __name__ == "__main__":
    main()
