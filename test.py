import torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from torch import optim

class QuantumEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, nhead, num_layers):
        super(QuantumEncoder, self).__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, nhead), num_layers)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)
        return x

class QuantumDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, nhead, num_layers):
        super(QuantumDecoder, self).__init__()
        self.embed = nn.Linear(output_dim, embed_dim)
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(embed_dim, nhead), num_layers)
        self.predict = nn.Linear(embed_dim, output_dim)

    def forward(self, x, context):
        x = self.embed(x)
        x = self.transformer_decoder(x.unsqueeze(0), context.unsqueeze(0)).squeeze(0)
        x = self.predict(x)
        return x

class QuantumTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, nhead, num_layers):
        super(QuantumTransformer, self).__init__()
        self.encoder = QuantumEncoder(input_dim, embed_dim, nhead, num_layers)
        self.decoder = QuantumDecoder(output_dim, embed_dim, nhead, num_layers)
        
    def forward(self, x, y):
        context = self.encoder(x)
        out = self.decoder(y, context)
        return out

def evaluate_model(model, dataloader):
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # set the model to evaluation mode

    total_loss = 0
    for i, batch in enumerate(dataloader):
        rho, measurement_record, optimal_delta = [b.to(device) for b in batch]

        # Shift the optimal_delta to create input and target sequences
        zeros = torch.zeros(optimal_delta.shape[0], 1).to(device)  # Create a tensor of zeros
        decoder_input = torch.cat((zeros, optimal_delta[:, :-1]), dim=1)  # Append the tensor of zeros at the beginning
        decoder_target = optimal_delta  # No need to shift decoder_target

        with torch.no_grad():  # no need to compute gradients during evaluation
            outputs = model(rho, decoder_input)  # Feed the shifted optimal_delta into the model
            loss = criterion(outputs, decoder_target)  # Compute loss between outputs and the shifted targets
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Evaluation Loss: {avg_loss}')
    wandb.log({"Test Loss": avg_loss})

    return avg_loss

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



model = torch.load("mymodel.pt")

dataset= QuantumDataset("markov_transformer/dataset_complete_delta_traj_seperable_imag_real_small_tunneling_total.pkl")
test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
evaluate_model(model, test_dataloader)
wandb.finish()


