from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, input_data, labels):
        self.input_data = input_data
        self.labels = labels

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.labels[idx]
