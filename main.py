import sys
from DatasetClasses import RnaDataset, SelexDataset

one_hot_mapping = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1]
}

label_mapping = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}


def get_selex_datasets(arguments, mapping):
    arguments = arguments[2:]  # keep selex files only
    selex_datasets = []
    for arg in arguments:
        selex_datasets.append(SelexDataset(arg, mapping))
    return selex_datasets


def main():
    arguments = sys.argv[1:]
    output_file = arguments[0]
    rna_dataset = RnaDataset(arguments[1], one_hot_mapping)
    # load selex files into an array of (unlabeled) SelexDataset objects, using one-hot mapping:
    selex_datasets = get_selex_datasets(arguments, one_hot_mapping)

    # examples:
    print(rna_dataset.sequences[0])
    print(selex_datasets[0][0])


if __name__ == "__main__":
    main()
