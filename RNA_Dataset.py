import torch
from torch.utils.data import Dataset


def sequence_to_tensor(sequence, mapping, data_type=torch.long):  # dtype corresponding to mapping
    return torch.tensor([mapping[base] for base in sequence], dtype=data_type)


class RnaDataset(Dataset):
    def __init__(self, file_path, mapping):
        self.file_path = file_path
        self.mapping = mapping
        self.sequences = self._load_sequences()
        self.length = len(self.sequences)  # Cache the length for efficiency

    def _load_sequences(self):
        sequences = []
        with open(self.file_path, 'r') as file:
            for line in file:
                if 'N' in line:
                    continue  # ignore sequences with 'N'
                sequence = line.strip()
                sequences.append(sequence_to_tensor(sequence, self.mapping))
        return sequences

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.sequences[index]
