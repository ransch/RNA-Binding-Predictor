import sys

import numpy as np


def _read_array_from_file(file_path):
    """
    Read a list from a given file.

    Args:
        file_path: The path of the file that should be parsed.

    Returns:
        The parsed list.
    """
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]


def main():
    # Parse the two files.
    list1 = _read_array_from_file(sys.argv[1])
    list2 = _read_array_from_file(sys.argv[2])

    # Ensure that the two lists are of the same length.
    if len(list1) != len(list2):
        raise ValueError('The given files must have the same number of lines.')

    # Calculate the Pearson correlation between the two lists.
    correlation = np.corrcoef(list1, list2)[0, 1]

    print(f'Pearson correlation: {correlation:.4f}')


if __name__ == '__main__':
    main()
