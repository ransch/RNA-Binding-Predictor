import tensorflow as tf

from dna_dataset_utils import sequence_to_tensor, random_sequence_dataset, _NUCLEOTIDES

_SHUFFLE_BUFFER_SIZE = 2000
_SELEX_SEQ_LEN = 40


def _parse_selex_line(line):
    """
    Parse a single SELEX line.

    Args:
        line: The line to parse.

    Returns:
         The tensor that represents the given line, or None if the line is invalid.
    """
    if 'N' in line:
        return None

    # Ignore the number of occurrences of the sequence.
    sequence, _ = line.strip().split(',', 1)
    return sequence_to_tensor(sequence)


def _selex_dataset_gen(file_path):
    """
    Generate sequences of a SELEX dataset of a single protein and a single experiment cycle.

    Args:
        file_path: The path to the SELEX dataset.

    Yields:
         The parsed tensors.
    """
    def _helper():
        with open(file_path, 'r') as file:
            for line in file:
                tensor = _parse_selex_line(line)
                if tensor is not None:
                    yield tensor
    return _helper


def _selex_dataset(file_path, cycle):
    output_signature = tf.TensorSpec(shape=(_SELEX_SEQ_LEN,len(_NUCLEOTIDES)), dtype=tf.int32)
    return tf.data.Dataset.zip((tf.data.Dataset.from_generator(_selex_dataset_gen(file_path), output_signature=output_signature),
                                tf.data.Dataset.from_tensor_slices(tf.constant([cycle])).repeat()))


_ZeroCycleDataset = tf.data.Dataset.zip((random_sequence_dataset(_SELEX_SEQ_LEN),
                                         tf.data.Dataset.from_tensor_slices(tf.constant([0])).repeat()))


def combined_selex_dataset(file_paths):
    """
    Create a SELEX dataset of all the experiment cycles of a single protein.
    The batches of the returned dataset are balanced.

    Args:
        file_paths: The paths of all the cycles of a single protein.

    Yields:
         The combined dataset.
    """
    # Create a shuffled dataset for each cycle.
    datasets = [
        _selex_dataset(file_path, cycle=index+1).shuffle(_SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        for index, file_path in enumerate(file_paths)
    ]
    datasets.append(_ZeroCycleDataset)

    # Make every dataset infinite.
    datasets = [dataset.repeat() for dataset in datasets]

    # Create the combined dataset. Each dataset is picked uniformly.
    return tf.data.Dataset.sample_from_datasets(datasets, rerandomize_each_iteration=True)
