import pytorch_lightning as pl
import torch
from kmeans_pytorch import kmeans
from torch import nn
from torch.nn import functional as F

from backend.models import DEVICE


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(24, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 5, 384),
            nn.ReLU(),
            nn.Linear(384, 5)
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 384),
            nn.ReLU(),
            nn.Linear(384, 512 * 5),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 5)),
            nn.ConvTranspose1d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 24, 3, padding=1),
        )

    def forward(self, x):
        return self.model(x)


class Autoencoder(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.lr = lr
        self.cuda = DEVICE

    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def kmeans_loss(self, emb):
        cluster_indices, cluster_centers = kmeans(X=emb, num_clusters=3, device=self.cuda)

        # Moving clusters to gpu, because library was made by clowns
        cluster_indices, cluster_centers = cluster_indices.to(self.cuda), cluster_centers.to(self.cuda)

        # Compute the distances between each data point and its assigned cluster center
        distances = torch.sum((emb - cluster_centers[cluster_indices]) ** 2, dim=1)

        # Compute the inertia (sum of squared distances)
        inertia = torch.sum(distances)
        return inertia

    def training_step(self, train_batch, batch_idx):
        x = train_batch
        emb = self(x)
        x_hat = self.decoder(emb)
        loss = F.mse_loss(x_hat, x)
        loss_kmeans = self.kmeans_loss(emb)
        loss = loss + 0.1 * loss_kmeans
        self.log('Training MSE loss', loss, on_step=False, on_epoch=True)        
        self.log('Training kmeans loss', loss_kmeans, on_step=False, on_epoch=True)
        # self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        emb = self(x)
        x_hat = self.decoder(emb)
        loss = F.mse_loss(x_hat, x)
        loss_kmeans = self.kmeans_loss(emb)
        loss = loss + 0.1 * loss_kmeans
        self.log('Validation MSE loss', loss, on_step=False, on_epoch=True)        
        self.log('Validation kmeans loss', loss_kmeans, on_step=False, on_epoch=True)
        return loss
