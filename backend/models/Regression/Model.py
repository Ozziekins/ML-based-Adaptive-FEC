from torch.nn import functional as F
import torch
from torch import nn
import pytorch_lightning as pl


class LossRatePredictor(pl.LightningModule):
    def __init__(self, n_features, hidden_dim, n_layers, output_size, learning_rate=0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.sequence = nn.Sequential(
            nn.Dropout(0.25),
            nn.BatchNorm1d(hidden_dim, affine=False),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        return self.sequence(hidden) * 100

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        # self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        # self.log('val_loss', loss, prog_bar=True)
        return loss
