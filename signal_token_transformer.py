import torch, os, sys
sys.path.append(os.getcwd())

import math
from torch import nn
from einops import rearrange, repeat
from swiss_library import initialize_weights

class GRUEncoder(nn.Module):
    def __init__(self, gru_hid_dim, gru_input_size, gru_layers, gru_dropout, bidirectional):
        super(GRUEncoder, self).__init__()
        
        self.fc = nn.Linear(gru_hid_dim*2, gru_hid_dim)
        self.attn_fc = nn.Linear(gru_hid_dim*2, gru_hid_dim*2)

        self.gru = nn.GRU(gru_input_size, gru_hid_dim, 
                            gru_layers, batch_first=True,
                            dropout=gru_dropout, bidirectional=bidirectional)
        
    def forward(self, x):
        """x : B, L, input_dim"""
        gru_out, _ = self.gru(x) # B, L, input_dim -> B, L, 2*H
        
        h_n = gru_out[:,-1,:] # B, 2*H
        signal_emb = self.fc(h_n) # B, 2*H -> B, H
            
        return signal_emb


class Multi_GRU(nn.Module):
    def __init__(self, gru_hid_dim, gru_input_size, gru_layers, gru_dropout, bidirectional, num_signals, emb_dim, device):
        super(Multi_GRU, self).__init__()

        self.num_signals = num_signals
        self.emb_dim = emb_dim
        self.device = device
        self.gru_input_size = gru_input_size
        self.gru_emb = GRUEncoder(gru_hid_dim, gru_input_size, gru_layers, gru_dropout, bidirectional)
        
        self.gru_dict = nn.ModuleDict()
        for i in range(num_signals):
            self.gru_dict[str(i)] = self.gru_emb
            
    def forward(self, x):
        """x : B, S, L"""
        B, S, _ = x.shape
        
        signals = torch.chunk(x, self.num_signals, dim=1) # B, S, L -> B, 1, L
        emb_out = torch.zeros(size=(B, S, self.emb_dim), 
                              device=self.device) # B, S, H
        
        for i, signal in enumerate(signals):
            # B, 1, L -> B, -1, gru input dim
            
            signal = signal.view(B, -1, self.gru_input_size).to(self.device)
            
            # B, -1, gru input dim -> B, H
            signal_emb = self.gru_dict[str(i)](signal)
            emb_out[:,i,:] = signal_emb
            
        return emb_out
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn # function to apply after normalization

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        """
        x : (batch, 19, dim)"""

        # (batch, 19, dim) -> (batch, 19, inner_dim) * 3
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        # (batch, heads, 19, head_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # (batch, heads, 19, 19) : attention score
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # (batch, heads, 19, 19) : attention score soft max
        attn = self.attend(dots)

        # (batch, heads, 19, head_dim) : softmax값에 value를 곱하여 z 생성
        out = torch.matmul(attn, v)

        # (batch, heads, 19, head_dim) -> (batch, 19, inner_dim) : z들을 concat
        out = rearrange(out, 'b h n d -> b n (h d)')

        """
        return (batch, 19, inner_dim) -> (batch, 19, dim)"""
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers: # type: ignore
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class STT(nn.Module):
    def __init__(self, gru_hid_dim, gru_input_size, gru_layers, gru_dropout, bidirectional, num_signals, emb_dim, device, emb_dropout, depth, 
                 heads, head_dim, transformer_mlp_dim, dropout, signal_emb):
        super().__init__()

        self.embedding_layers = Multi_GRU(gru_hid_dim, gru_input_size, gru_layers, gru_dropout, bidirectional, num_signals, emb_dim, device)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.transformer_encoder = Transformer(emb_dim, depth, heads, head_dim, 
                                               transformer_mlp_dim, dropout)

        self.sensor_emb = nn.Parameter(torch.randn(1, num_signals, emb_dim), requires_grad=True)
        self.signal_emb = signal_emb
            
        self.apply(initialize_weights)
        
    def forward(self, signals):
        """
        signals shape : (batch, # of signals, data length)"""

        x = self.embedding_layers(signals) # x : (batch, # of signals, emb_dim)
        b, _, _ = x.shape

        if self.signal_emb:
            # sensor_tokens : (1, num_signals, dim) -> (batch, num_signals, dim)
            sensor_embedding = repeat(self.sensor_emb, '() n d -> b n d', b = b)
            
            # x : (batch, # of signals, dim) -> (batch, num_signals, dim)
            x += sensor_embedding
        
        x = self.emb_dropout(x)

        # x : (batch, num_signals, dim) -> (batch, num_signals, dim)
        enc_out = self.transformer_encoder(x)
        mean_out = enc_out.mean(dim=1) # batch, dim
        
        features = {'enc_out':enc_out, 'mean_out':mean_out}
        
        return features