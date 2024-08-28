# Predicting RNACompete binding intensities from HTR-SELEX data

Students:
- Amir Schreiber
- Ran Schreiber

This repository contains the source code for the final project in Deep Learning in Computational
Biology course.

- `main.py` - The main Python script. It accepts the path of the output file, an RNAcompete sequence
  file, and paths of different cycles of HTR-SELEX (of a single RBP). It writes the predicted
  binding intensities into the output file. The model's properties and the training progress are
  displayed, and model's weights and training history are optionally saved (according to the
  `_SHOULD_SAVE_MODEL` global).
- `compute_correlation.py` - A Python script that computes the Pearson correlation of two vectors.
- `plot_history_graphs.py` - A Python script that loads saved history from a file, and creates
  two graphs, one with the progress of the accuracy values of the training and validation sets, and
  the second with the progress of the loss values of the training and validation sets.
- `get_predictions_from_saved_model.py` - A script for loading a saved model and getting predicted
  binding intensities.

The scripts require Keras, TensorFlow, Numpy, and Matplotlib - see `requirements.txt`.
