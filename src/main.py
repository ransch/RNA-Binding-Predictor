import sys
from selex_dataset import combined_selex_dataset
from dna_dataset_utils import augment_dataset

def main():
    output_file_path = sys.argv[1]
    rna_file_path = sys.argv[2]
    selex_file_paths = sys.argv[3:]

    selex_ds = augment_dataset(combined_selex_dataset(selex_file_paths))



if __name__ == "__main__":
    main()
