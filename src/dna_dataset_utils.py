import tensorflow as tf

_NUCLEOTIDES = tf.constant(['A', 'C', 'G', 'T'], dtype=tf.string)
_AUGMENTATION_PROB = 0.7
_MUTATION_FRAC = 0.2
_INSERTION_LEN = 5
_DELETION_LEN_MIN = 1
_DELETION_LEN_MAX = 8

# A static mapping between nucleotides and their indices.
_NUCLEOTIDE_TO_INDEX = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(_NUCLEOTIDES),
        values=tf.constant(range(len(_NUCLEOTIDES)), dtype=tf.int32)
    ),
    default_value=tf.constant(-1)
)


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
    indices = _NUCLEOTIDE_TO_INDEX.lookup(tf.strings.bytes_split(sequence))
    return tf.one_hot(indices, len(_NUCLEOTIDES), dtype=tf.int32)


def dna_line_filter(line):
    """
    A filter of valid lines with DNA sequences.

    Args:
        line: The line that should be validated.

    Returns:
        Whether the line is valid.
    """
    return tf.strings.regex_full_match(line, "^[^N]*$")


def _random_dna_sequence(seq_len):
    """
    Return a random DNA sequence of the given length.

    Args:
        seq_len: The length of the returned DNA sequence.

    Returns:
        A tensor that represents a random DNA sequence of the given length.
    """
    indices = tf.random.uniform([seq_len], 0, len(_NUCLEOTIDES), dtype=tf.int32)
    return tf.one_hot(indices, len(_NUCLEOTIDES), dtype=tf.int32)


def random_sequence_dataset(seq_len):
    """
    Create an infinite dataset consisting of random DNA sequences of the given length.

    Args:
        dataset_size: The size of the dataset.
        seq_len: The length of the sequences in the dataset.

    Returns:
        A dataset consisting of random DNA sequences of the given length.
    """

    def _helper():
        while True:
            yield _random_dna_sequence(seq_len)

    output_signature = tf.TensorSpec(shape=(seq_len, len(_NUCLEOTIDES)), dtype=tf.int32)
    return tf.data.Dataset.from_generator(_helper, output_signature=output_signature)


def _augmentation_mutation(sequence):
    """
    Randomly change the values of a few indices in the given sequence.

    Args:
        sequence: A tensor that represents a DNA sequence.

    Returns:
        The modified sequence.
    """
    # Randomly select _MUTATION_FRAC indices.
    sequence_len = tf.cast(tf.shape(sequence)[0], tf.float32)
    changed_nucleotides_count = tf.cast(tf.floor(tf.multiply(sequence_len, _MUTATION_FRAC)),
                                        tf.int32)
    indices_to_replace = tf.random.shuffle(tf.range(sequence_len, dtype=tf.int32))[
                         :changed_nucleotides_count]

    # Randomly determine the replacement of the selected indices.
    random_nucleotide_indices = tf.random.uniform([changed_nucleotides_count], 0, len(_NUCLEOTIDES),
                                                  dtype=tf.int32)
    replacement = tf.one_hot(random_nucleotide_indices, depth=len(_NUCLEOTIDES), dtype=tf.int32)

    return tf.tensor_scatter_nd_update(tensor=sequence,
                                       indices=tf.expand_dims(indices_to_replace, 1),
                                       updates=replacement)


def _augmentation_translocation(sequence):
    """
    Split the given sequence at a random index and swap the two parts.

    Args:
        sequence: A tensor that represents a DNA sequence.

    Returns:
        The modified sequence.
    """
    split_index = tf.random.uniform([], minval=0, maxval=len(_NUCLEOTIDES) - 1, dtype=tf.int32)
    return tf.concat([sequence[split_index:], sequence[:split_index]], axis=0)


def _augmentation_insertion(sequence):
    """
    Insert a random subsequence into the given sequence.

    Args:
        sequence: A tensor that represents a DNA sequence.

    Returns:
        The modified sequence.
    """
    new_part = _random_dna_sequence(_INSERTION_LEN)
    insertion_index = tf.random.uniform([], minval=0, maxval=len(sequence), dtype=tf.int32)
    return tf.concat([sequence[:insertion_index], new_part, sequence[insertion_index:]], axis=0)


def _augmentation_deletion(sequence):
    """
    Delete a random subsequence from the given sequence.

    Args:
        sequence: A tensor that represents a DNA sequence.

    Returns:
        The modified sequence.
    """
    deletion_length = tf.random.uniform([], minval=_DELETION_LEN_MIN, maxval=_DELETION_LEN_MAX,
                                        dtype=tf.int32)
    deletion_index = tf.random.uniform([], minval=0, maxval=len(sequence) - deletion_length,
                                       dtype=tf.int32)
    return tf.concat([sequence[:deletion_index], sequence[deletion_index + deletion_length:]],
                     axis=0)


_DNA_AUGMENTATION_METHODS = [_augmentation_mutation, _augmentation_translocation,
                             _augmentation_insertion, _augmentation_deletion]


def _randomly_augment_sequence(sequence):
    """
    Apply a random augmentation from `_DNA_AUGMENTATION_METHODS` to the given DNA sequence.

    Args:
        sequence: A tensor that represents a DNA sequence.

    Returns:
        A random augmentation of the given DNA sequence.
    """
    random_index = tf.random.uniform([], minval=0, maxval=len(_DNA_AUGMENTATION_METHODS),
                                     dtype=tf.int32)

    cases = {
        i: lambda i=i: _DNA_AUGMENTATION_METHODS[i](sequence)
        for i in range(len(_DNA_AUGMENTATION_METHODS))
    }

    return tf.switch_case(random_index, cases)


def _randomly_augment_sequence_decision(sequence):
    """
    Decide randomly if the given DNA sequence should be modified. If it should, apply
    `_randomly_augment_sequence`. Otherwise, return the non-modified sequence.

    Args:
        sequence: A tensor that represents a DNA sequence.

    Returns:
        A random augmentation of the given DNA sequence.
    """
    random_number = tf.random.uniform([], 0, 1)

    # Call _randomly_augment_sequence with probability _AUGMENTATION_PROB.
    result = tf.cond(
        random_number >= _AUGMENTATION_PROB,
        true_fn=lambda: _randomly_augment_sequence(sequence),
        false_fn=lambda: sequence
    )

    return result


def augment_dataset(dataset):
    """
    Randomly perform several data-augmentation techniques on the given dataset.

    Args:
        dataset: The dataset to be augmented.

    Returns:
        The modified dataset.
    """
    return dataset.map(
        lambda sequence, label: (_randomly_augment_sequence_decision(sequence), label))
