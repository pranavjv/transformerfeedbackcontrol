from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle


class QuantumDataset(Dataset):
    def __init__(self, dataset_path, padding_value=0):
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        self.padding_value = padding_value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rho, measurement_record, optimal_delta = self.dataset[idx]

        # Pad measurement_record to max length in the dataset
        max_len = max(len(record) for _, record, _ in self.dataset)
        #print("measurement record shape: ", len(measurement_record))
        #print("measurement before padding ", measurement_record[0])
        measurement_record = np.pad(measurement_record, (0, max_len - len(measurement_record)), constant_values=self.padding_value)
        #print("measurement record shape after padding: ", len(measurement_record))
        #print("measurement after padding", measurement_record)

        # Stack rho and measurement_record as the input, and optimal_delta as the target
        return torch.from_numpy(np.hstack((rho, measurement_record)).astype(np.float32)), torch.tensor([optimal_delta], dtype=torch.float32), torch.from_numpy(rho).float()


# Load the dataset
dataset = QuantumDataset('markov_transformer/dataset_complete_delta_traj_seperable_imag_real_ten_thousand_complete_traj.pkl')

# Split the dataset into training and test sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#print(train_dataset[0][0].shape)
