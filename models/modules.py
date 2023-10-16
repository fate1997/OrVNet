import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import global_add_pool, global_max_pool


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, dropout, activation):
        super().__init__()
        if layers == 1:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            self.layers = []
            for i in range(layers - 1):
                self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
                self.layers.append(activation)
                self.layers.append(nn.LayerNorm(hidden_dim))
                self.layers.append(nn.Dropout(dropout))

            self.layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.Sequential(*self.layers)
        
        self.layers.apply(init_weight)
        
    def forward(self, x):
        output = self.layers(x)
        return output


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias != None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        
class PositionEncoder(nn.Module):
    def __init__(self, d_model, seq_len=4, device='cuda:0'):
        super().__init__()
        # position_enc.shape = [seq_len, d_model]
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] for pos in range(seq_len)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        self.position_enc = torch.tensor(position_enc, device=device).unsqueeze(0).float()

    def forward(self, x):
        # x.shape = [batch_size, seq_length, d_model]
        x = x * Variable(self.position_enc, requires_grad=False)
        return x


class ReadoutPhase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # readout phase
        self.weighting = nn.Linear(dim, 1) 
        self.score = nn.Sigmoid() 
        
        nn.init.xavier_uniform_(self.weighting.weight)
        nn.init.constant_(self.weighting.bias, 0)
    
    def forward(self, x, batch):
        weighted = self.weighting(x)
        score = self.score(weighted)
        output1 = global_add_pool(score * x, batch)
        output2 = global_max_pool(x, batch)
        
        output = torch.cat([output1, output2], dim=1)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(input_dim, d_model)
        self.wk = nn.Linear(input_dim, d_model)
        self.wv = nn.Linear(input_dim, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if input_dim == d_model:
            self.residual_linear = nn.Identity()
        else:
            self.residual_linear = nn.Linear(input_dim, d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.zeros_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x (input features): [num_nodes, input_dim]
            mask (mask matrix): [num_heads, num_nodes, num_nodes]
        """
        bs = x.shape[0]
        residual = self.residual_linear(x)
        q = self.wq(x).view(bs, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(x).view(bs, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(x).view(bs, -1, self.num_heads, self.depth).transpose(1, 2)
        
        # shape of attn = [num_heads, num_nodes, num_nodes]
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.depth)  
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # [num_nodes, num_heads * depth]
        output = torch.matmul(attn, v).transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.dense(output)
        output  = self.ln(residual + output)
        
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_dim, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(model_dim, d_ff)
        self.w2 = nn.Linear(d_ff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(model_dim)
        self.act = nn.ELU()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
        nn.init.zeros_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
        
    def forward(self, x):
        output = self.act(self.w1(x))
        output = self.dropout(output)

        output = self.ln(x + self.dropout(self.w2(output)))
        return output



class BoltzmannLayer(nn.Module):
    def __init__(self, dim=128, num_levels=16):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.cal_energies = MLP(self.dim, 256, self.num_levels * 2, 3, 0.2, nn.ELU())
        self.cal_level_prob = nn.Softmax(dim=-1)
    
    def init_params(self):
        self.cal_energies.apply(init_weight)
    
    def forward(self, x, temperature):
        energies = self.cal_energies(x)

        energies_double = energies.view(-1, 2, self.num_levels)
        energies = energies_double[:, 0, :]
        energies_value = energies_double[:, 1, :]
        coef = (1 / (8.314 * 0.001 * temperature)) # [batch_size, num_temperature]
        exponent = -1 *  torch.matmul(coef.unsqueeze(-1), energies.unsqueeze(1))
        exponent = exponent.view(-1, temperature.size(1), self.num_levels)
        prob = self.cal_level_prob(exponent)
        
        energy = (prob *energies_value.view(-1, 1, self.num_levels))

        return energy


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, heads, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.attn = MultiHeadAttention(input_dim, d_model, heads, dropout=0.1)
        self.ff = PositionWiseFeedForward(d_model, d_model, dropout=dropout)

    def forward(self, x):
        if self.input_dim == self.d_model:
            x1 = self.attn(x) + x
        else:
            x1 = self.attn(x)
        x2 = x1 + self.ff(x1)
        return x2

