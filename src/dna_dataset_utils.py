import random

import numpy as np
import tensorflow as tf

_NUCLEOTIDES = ['A', 'C', 'G', 'T']
_AUGMENTATION_PROB = 0.7
_MUTATION_FRAC = 0.1
_INSERTION_LEN = 5


def sequence_to_tensor(sequence):
    """
    Convert a string DNA sequence into a TensorFlow tensor.
    The tensor's shape is (n, 4), where n is the length of the sequence.
    Each row in the tensor is the one-hot representation of the corresponding nucleotide.

    Args:
        sequence: The sequence to be converted to a tensor.

    Returns:
        The tensor that represents the given sequence.
    """
    return tf.one_hot([_NUCLEOTIDES.index(c) for c in sequence], len(_NUCLEOTIDES))


def _random_dna_sequence(seq_len):
    """
    Return a random DNA sequence of the given length.

    Args:
        seq_len: The length of the returned DNA sequence.

    Returns:
        A tensor that represents a random DNA sequence of the given length.
    """
    indices = sequence_to_tensor(
        tf.random.uniform([seq_len], 1, seq_len(_NUCLEOTIDES) + 1, dtype=tf.int32))
    return tf.one_hot(indices, len(_NUCLEOTIDES))


def random_sequence_dataset(seq_len):
    """
    Create an infinite dataset consisting of random DNA sequences of the given length.

    Returns:
        A dataset consisting of random DNA sequences of the given length.
    """

    def _helper():
        while True:
            yield _random_dna_sequence(seq_len)

    return tf.data.Dataset.from_generator(_helper())


def _augmentation_mutation(sequence):
    sequence_len = tf.shape(sequence)[0].numpy()
    changed_nucleotides_count = int(sequence_len * _MUTATION_FRAC)

    indices_to_replace = tf.random.shuffle(tf.range(sequence_len))[:changed_nucleotides_count]

    random_nucleotide_indices = tf.random.uniform([changed_nucleotides_count], 0, len(_NUCLEOTIDES),
                                                  dtype=tf.int32)
    replacement = tf.one_hot(random_nucleotide_indices, depth=len(_NUCLEOTIDES), dtype=tf.int32)

    return tf.tensor_scatter_nd_update(sequence, tf.expand_dims(indices_to_replace, 1),
                                       replacement)


def _augmentation_translocation(sequence):
    index = random.randint(0, len(_NUCLEOTIDES) - 1)
    if index == 0:
        return sequence
    return tf.concat([sequence[index:], sequence[:index]], axis=0)


def _augmentation_insertion(sequence):
    new_part = _random_dna_sequence(_INSERTION_LEN)
    return tf.concat([sequence[:_INSERTION_LEN], new_part, sequence[_INSERTION_LEN:]], axis=0)


_DNA_AUGMENTATION_METHODS = [_augmentation_mutation, _augmentation_translocation,
                             _augmentation_insertion]


def _augment_sequence(sequence, label):
    """
    Randomly augment a DNA sequence.

    Args:
        sequence: A tensor that represents a DNA sequence.

    Returns:
        A random augmentation of the given DNA sequence.
    """
    if np.random.binomial(1, 1 - _AUGMENTATION_PROB):
        return sequence

    return random.choice(_DNA_AUGMENTATION_METHODS)(sequence), label


def augment_dataset(dataset):
    """
    Randomly Perform several data-augmentation techniques on the given dataset.

    Args:
        dataset: The dataset to be augmented.
    :return:
    """
    return dataset.map(lambda sequence: _augment_sequence(sequence))
