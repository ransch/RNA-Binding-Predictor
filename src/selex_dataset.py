import tensorflow as tf

from dna_dataset_utils import random_sequence_dataset, dna_line_filter, sequence_to_tensor

_SHUFFLE_BUFFER_SIZE = 100000
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
        (ds, tf.data.Dataset.from_tensor_slices(tf.constant([cycle])).repeat()))


# An infinite dataset of the zero cycle.
_ZeroCycleDataset = tf.data.Dataset.zip((random_sequence_dataset(_SELEX_SEQ_LEN),
                                         tf.data.Dataset.from_tensor_slices(
                                             tf.constant([0])).repeat()))


def combined_selex_dataset(file_paths, validation_data_size):
    """
    Create a SELEX dataset of all the experiment cycles of a single protein.
    The batches of the returned dataset are balanced.

    Args:
        file_paths: The paths of all the cycles of a single protein. It's a dictionary that maps
                    the cycle index to its path.
        validation_data_size: The size of the validation set.

    Returns:
         A pair of the training and validation sets. The training set is infinite, and the
         validation set consists of `validation_data_size` examples.
    """
    # The number of examples of each cycle in the returned validation set.
    each_cycle_val_count = validation_data_size // (len(file_paths) + 1)

    # Create a dataset for each positive cycle.
    datasets = [
        _selex_dataset(file_path, cycle) for cycle, file_path in file_paths.items()
    ]

    # Split the datasets into training and validation sets.
    val_ds = [
        ds.take(each_cycle_val_count) for ds in datasets
    ]
    train_ds = [
        ds.skip(each_cycle_val_count) for ds in datasets
    ]

    # Shuffle the validation sets.
    val_ds = [ds.shuffle(validation_data_size, reshuffle_each_iteration=True) for ds in val_ds]
    # Make the training sets infinite and shuffle them.
    train_ds = [ds.repeat().shuffle(_SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
                for ds in train_ds]

    # Split the zero cycle dataset into training and validation sets.
    zero_cycle_val_ds = _ZeroCycleDataset.take(each_cycle_val_count)
    zero_cycle_train_ds = _ZeroCycleDataset.skip(each_cycle_val_count)

    val_ds.append(zero_cycle_val_ds)
    train_ds.append(zero_cycle_train_ds)

    # Create the combined train and validation sets. Each cycle dataset is picked uniformly.
    return (tf.data.Dataset.sample_from_datasets(train_ds, rerandomize_each_iteration=True),
            tf.data.Dataset.sample_from_datasets(val_ds, rerandomize_each_iteration=True))
