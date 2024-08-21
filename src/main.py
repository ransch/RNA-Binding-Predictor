import sys
import tensorflow as tf
import keras
from keras import layers
from keras import regularizers

from dna_dataset_utils import augment_dataset
from rna_compete_dataset import rna_compete_dataset
from selex_dataset import combined_selex_dataset

_BATCH_SIZE = 32
_MAX_EPOCHS_NUM = 5000
_VALIDATION_SPLIT = 0.2
_MIN_DELTA = 0.1
_PATIENCE = 50


def _get_model():
    model = keras.Sequential()

    # 2-layered LSTM with Batch Normalization
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.BatchNormalization())  # Batch Normalization after the first LSTM layer

    model.add(layers.LSTM(64))
    model.add(layers.BatchNormalization())  # Batch Normalization after the second LSTM layer

    # Fully connected layers with Batch Normalization
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())  # Batch Normalization before the output layer

    model.add(layers.Dense(4, activation='softmax'))  # Output layer with 4 units for classification

    # Compile the model
    model.compile(optimizer='adam',
                  loss='SparseCategoricalCrossentropy',
                  metrics=['accuracy'])

    return model


def _results_to_binding_predictions(model_results):
    pass  # TODO


def _write_predictions(output_file_path, predictions):
    pass  # TODO


def main():
    output_file_path = sys.argv[1]
    rna_compete_file_path = sys.argv[2]
    selex_file_paths = sys.argv[3:]

    selex_ds = augment_dataset(combined_selex_dataset(selex_file_paths)).batch(_BATCH_SIZE)
    model = _get_model()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                   min_delta=_MIN_DELTA, patience=_PATIENCE)

    # Manually split dataset into training and validation sets
    total_size = (tf.data.experimental.cardinality(selex_ds)).numpy()
    val_size = int(total_size * _VALIDATION_SPLIT)
    train_size = total_size - val_size
    train_ds = selex_ds.take(train_size)
    val_ds = selex_ds.skip(train_size).take(val_size)
    model.fit(train_ds,
              epochs=_MAX_EPOCHS_NUM,
              callbacks=[early_stopping],
              validation_data=val_ds)

    rna_compete_ds = rna_compete_dataset(rna_compete_file_path)
    model_results = model.predict(rna_compete_ds)
    predictions = _results_to_binding_predictions(model_results)

    _write_predictions(output_file_path, predictions)


if __name__ == "__main__":
    main()
