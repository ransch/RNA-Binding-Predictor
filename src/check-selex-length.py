import sys
import os


def main():
    selex_files_path = sys.argv[1]
    file_num = 0
    flag = 1
    for filename in os.listdir(selex_files_path):
        file_num += 1
        filepath = os.path.join(selex_files_path, filename)
        with open(filepath, 'r') as file:
            for line in file:
                sequence, _ = line.strip().split(',', 1)
                if len(sequence) != 40:
                    print(sequence)
                    flag = 0
    print("Processed", file_num, "files.")
    if flag == 1:
        print("All selex sequences are of length 40.")


if __name__ == "__main__":
    main()
