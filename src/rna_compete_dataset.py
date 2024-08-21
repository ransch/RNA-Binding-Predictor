import tensorflow as tf

from dna_dataset_utils import sequence_to_tensor


def _rna_compete_dataset_gen(file_path):
    """
    Generate sequences of an RNAcompete dataset of a single protein.

    Args:
        file_path: The path to the RNAcompete dataset.

    Yields:
         The parsed tensors.
    """
    with open(file_path, 'r') as file:
        for line in file:
            tensor = sequence_to_tensor(line.strip())
            if tensor is not None:
                yield tensor


def rna_compete_dataset(file_path):
    """
    Create an RNAcompete dataset of a single protein.

    Args:
        file_path: The path of the RNAcompete dataset.

    Returns:
        The parsed dataset.
    """
    # TODO change _rna_compete_dataset_gen to return a callable
    return tf.data.Dataset.from_generator(_rna_compete_dataset_gen(file_path))
