from torch import nn
from swiss_library import initialize_weights

class Projector(nn.Module):
    def __init__(self, emb_dim, num_signals, proj_hiddim, proj_dim):
        super().__init__()
        
        flatten_dim = emb_dim*num_signals

        self.norm = nn.BatchNorm1d(proj_hiddim)

        self.net = nn.Sequential(
            nn.Linear(flatten_dim, proj_hiddim),
            self.norm, nn.ReLU(inplace=True),
            nn.Linear(proj_hiddim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class Predictor(nn.Module):
    def __init__(self, proj_hiddim, proj_dim):
        super().__init__()
        
        self.norm = nn.BatchNorm1d(proj_hiddim)    
                
        self.net = nn.Sequential(
            nn.Linear(proj_dim, proj_hiddim),
            self.norm, nn.ReLU(inplace=True),
            nn.Linear(proj_hiddim, proj_dim)
        )
        self.apply(initialize_weights)
        
    def forward(self, x):
        return self.net(x)
    
class ReconstructionHead(nn.Module):
    def __init__(self, num_signals, input_dim, feature_dim):
        super(ReconstructionHead, self).__init__()
        
        self.num_signals = num_signals
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        self.recon = nn.Linear(self.num_signals*self.input_dim,
                            self.num_signals*self.feature_dim)
        self.apply(initialize_weights)
        
    def forward(self, x):
        """x shape : batch, signals, emb dim"""
        x = x.view(-1, self.num_signals*self.input_dim) # batch, num_signals*emb_dim
        x = self.recon(x) # batch, num_signals*feature_dim
        x = x.view(-1, self.num_signals, self.feature_dim) # batch, num_signals, feature dim
        
        return x