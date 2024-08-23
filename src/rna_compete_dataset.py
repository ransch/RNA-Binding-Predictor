import tensorflow as tf

from dna_dataset_utils import dna_line_filter, sequence_to_tensor


def rna_compete_dataset(file_path):
    """
    Create an RNAcompete dataset of a single protein.

    Args:
        file_path: The path of the RNAcompete dataset.

    Returns:
        An RNAcompete dataset.
    """
    ds = tf.data.TextLineDataset(file_path)
    # Filter out invalid lines.
    ds = ds.filter(dna_line_filter)
    # Parse the lines as tensors.
    ds = ds.map(sequence_to_tensor, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds
