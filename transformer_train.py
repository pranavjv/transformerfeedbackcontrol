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

def train_model(model, dataloader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            rho, measurement_record, optimal_delta = [b.to(device) for b in batch]
            optimizer.zero_grad()

            # Shift the optimal_delta to create input and target sequences
            zeros = torch.zeros(optimal_delta.shape[0], 1).to(device)  # Create a tensor of zeros
            decoder_input = torch.cat((zeros, optimal_delta[:, :-1]), dim=1)  # Append the tensor of zeros at the beginning
            decoder_target = optimal_delta  # No need to shift decoder_target
            
            outputs = model(rho, decoder_input)  # Feed the shifted optimal_delta into the model
            loss = criterion(outputs, decoder_target)  # Compute loss between outputs and the shifted targets
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs} Loss: {loss.item()}')
        wandb.log({"Train Loss": loss})


    return model

model= QuantumTransformer(102, 4, 128, 4, 2)

wandb.init(project="markovian_transformer")
wandb.watch(model)
model = train_model(model, train_dataloader, epochs=10, lr=0.001)
#save the model
torch.save(model, "mymodel.pt")
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'mymodelscript.pt')
print(model)
