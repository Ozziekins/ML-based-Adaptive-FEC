from torch import nn
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers, output_size):        
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.batch_norm = nn.BatchNorm1d(10, affine=False)
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)        
        self.sequence = nn.Sequential(
            nn.Dropout(0.25),
            nn.BatchNorm1d(hidden_dim, affine=False),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.25),    
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.Dropout(0.25),    
            nn.Linear(128, output_size),
            nn.Dropout(0.25),    
            nn.ReLU(),            
        )
    def forward(self, X_batch):        
        _, (hidden, _) = self.lstm(self.batch_norm(X_batch))
        hidden = hidden[-1]
        return  self.sequence(hidden)