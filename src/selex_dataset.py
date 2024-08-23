import tensorflow as tf

from dna_dataset_utils import random_sequence_dataset, dna_line_filter, sequence_to_tensor

_SHUFFLE_BUFFER_SIZE = 20000
_SELEX_SEQ_LEN = 40


def _parse_selex_line(line):
    """
    Parse a single SELEX line.

    Args:
        line: The line to parse.

    Returns:
         The tensor that represents the given line.
    """
    split_line = tf.strings.split(line, sep=',', maxsplit=1)
    sequence = split_line[0]
    return sequence_to_tensor(sequence)


def _selex_dataset(file_path, cycle):
    """
    A SELEX dataset of a single protein and a single experiment cycle.

    Args:
        file_path: The path to the SELEX dataset.
        cycle: The cycle that corresponds to the given file.

    Returns:
        A SELEX dataset.
    """
    ds = tf.data.TextLineDataset(file_path)
    # Filter out invalid lines.
    ds = ds.filter(dna_line_filter)
    # Parse the lines as tensors.
    ds = ds.map(_parse_selex_line, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return tf.data.Dataset.zip(
        (ds, tf.data.Dataset.from_tensor_slices(tf.constant([cycle])))).cache()


# An infinite dataset of the zero cycle.
_ZeroCycleDataset = tf.data.Dataset.zip((random_sequence_dataset(_SELEX_SEQ_LEN),
                                         tf.data.Dataset.from_tensor_slices(
                                             tf.constant([0])).repeat()))


def combined_selex_dataset(file_paths):
    """
    Create a SELEX dataset of all the experiment cycles of a single protein.
    The batches of the returned dataset are balanced.

    Args:
        file_paths: The paths of all the cycles of a single protein. It's a dictionary that maps
        the cycle index to its path.

    Yields:
         The combined dataset.
    """
    # Create a shuffled dataset for each cycle.
    datasets = [
        _selex_dataset(file_path, cycle).repeat().shuffle(_SHUFFLE_BUFFER_SIZE,
                                                          reshuffle_each_iteration=True)
        for cycle, file_path in file_paths.items()
    ]

    datasets.append(_ZeroCycleDataset)

    # Create the combined dataset. Each dataset is picked uniformly.
    return tf.data.Dataset.sample_from_datasets(datasets, rerandomize_each_iteration=True)
