import os.path
from tqdm import tqdm
from Regression.NeuralModel import LSTMRegressor
import torch

class LSTMPerCluster:
  def __init__(self, cluster_number,loss_fn, optimizer):
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.cluster_number = cluster_number
    self.file_path = f'model_{cluster_number}.pth'
    if os.path.isfile(self.file_path):
      self.model = torch.load(self.file_path)
    else:
      self.model = LSTMRegressor(6,8,5,5)

  def train(self, train_loader, epochs=10, save=True, device=None):
    for i in range(1, epochs+1):
      losses = []
      for X, Y in tqdm(train_loader):
        X, Y = X.to(device), Y.to(device)
        Y_preds = self.model(X)
        loss = self.loss_fn(Y_preds, Y)
        losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
    if save:
      torch.save(self.model, self.file_path)