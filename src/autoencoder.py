import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(X_train, epochs=20):
    model = Autoencoder(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X_train.values).float()

    for epoch in range(epochs):
        output = model(X_tensor)
        loss = loss_fn(output, X_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def reconstruction_error(model, X):
    X_tensor = torch.tensor(X.values).float()
    recon = model(X_tensor).detach().numpy()

    error = ((X.values - recon) ** 2).mean(axis=1)
    return error