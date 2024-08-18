import torch
from torch.utils.data import Dataset


def sequence_to_tensor(sequence, mapping):
    return torch.tensor([mapping[base] for base in sequence])


class BaseSequenceDataset(Dataset):
    def __init__(self, file_path, mapping):
        self.file_path = file_path
        self.mapping = mapping
        self.sequences = self._load_sequences()
        self.length = len(self.sequences)  # Cache the length for efficiency

    def _load_sequences(self):
        sequences = []
        with open(self.file_path, 'r') as file:
            for line in file:
                sequence = line.strip()
                if 'N' not in sequence:
                    sequences.append(sequence_to_tensor(sequence, self.mapping))
        return sequences

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.sequences[index]


class RnaDataset(BaseSequenceDataset):
    pass


class SelexDataset(BaseSequenceDataset):
    def _load_sequences(self):
        sequences = []
        with open(self.file_path, 'r') as file:
            for line in file:
                if 'N' not in line:
                    sequence, _ = line.strip().split(',', 1)  # get sequence (before comma)
                    sequences.append(sequence_to_tensor(sequence, self.mapping))
        return sequences
