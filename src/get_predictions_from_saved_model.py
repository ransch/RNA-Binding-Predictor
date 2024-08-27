import sys
from main import _get_model1, _get_model2, _get_model3, _get_model4, _get_model5, _results_to_binding_predictions, _write_predictions
from rna_compete_dataset import rna_compete_dataset

_LAST_DENSE_SIZE = 5
_PREDICTION_BATCH_SIZE = 512


def main():
    output_file_path = sys.argv[1]
    rna_compete_file_path = sys.argv[2]
    weights_file_path = sys.argv[3]

    model = _get_model2(_LAST_DENSE_SIZE)  # change model according to need
    model.load_weights(weights_file_path)

    rna_compete_ds = rna_compete_dataset(rna_compete_file_path).padded_batch(
        batch_size=_PREDICTION_BATCH_SIZE, padded_shapes=[45, 4], padding_values=0)

    model_results = model.predict(rna_compete_ds)
    predictions = _results_to_binding_predictions(model_results, labels=[0, 1, 2, 3, 4])
    _write_predictions(output_file_path, predictions)


if __name__ == "__main__":
    main()
