import torch
from torch import nn

from models.modules import MLP, TransformerEncoder
from models.OrVNet_FFiNet import OrVNetModel

from .FFiNet_model import FFiLayer
from .modules import MLP, BoltzmannLayer, ReadoutPhase


class Interaction(nn.Module):
    def __init__(self, in_dim, dim, heads, dropout, num_layers):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.repr_token = nn.Parameter(torch.FloatTensor(1, in_dim))
        self.encoders = nn.ModuleList([TransformerEncoder(in_dim, dim, heads, dropout) if i == 0 else TransformerEncoder(dim, dim, heads, dropout) for i in range(num_layers)])
    
    def forward(self, x1, x2):
        in_set = torch.stack([self.repr_token.expand(x1.size(0), self.in_dim), x1, x2], dim=1)
        for i, encoder in enumerate(self.encoders):
            if i == 0:
                output = encoder(in_set)
            else:
                output = encoder(output)
        return output[:, 0, :]


class OrVMixNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        

        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers  
        self.num_heads = config.model.num_heads
        self.dropout = config.model.dropout 
        self.num_energies = config.model.num_energies
        self.return_repr = config.model.return_repr
        self.pretrained_path = config.train.pretrained_path
        
        self.message_passing = nn.ModuleList()
        for i in range(self.num_layers):
            self.message_passing.append(
                FFiLayer(
                    self.input_dim if i == 0 else self.hidden_dim,
                    self.hidden_dim // self.num_heads,
                    self.num_heads,
                    dropout=self.dropout
                ))
        
        self.readout = ReadoutPhase(self.hidden_dim)

        self.pred_boltzmann = BoltzmannLayer(self.hidden_dim*2+2, self.num_energies)
        self.pred_surmount_prob = nn.Sigmoid()
        self.pred_extra = MLP(self.num_energies, 128, 1, 2, 0.1, nn.ELU())

        self.inter_module = Interaction(2*self.hidden_dim, 128, 8, 0.1, 3)

        # pretrain
        if self.pretrained_path != 'None':
            self.mol_repr_model = OrVNetModel(config)
            self.mol_repr_model.eval()
            print(self.mol_repr_model.load_state_dict(torch.load(config.train.pretrained_path)))
            for name, param in self.mol_repr_model.named_parameters():
                param.requires_grad = False
                    
    
    def forward_pure(self, data):
        if self.pretrained_path != 'None':
            _, mol_repr, ln_visc = self.mol_repr_model(data)
        else:
            x, edge_index = data.x, data.edge_index
            triple_index, quadra_index, pos, edge_attr= data.triple_index, data.quadra_index, data.pos, data.edge_attr
            for layer in self.message_passing:
                x = layer((x, edge_index, triple_index, quadra_index, pos, edge_attr))
            
            mol_repr = self.readout(x, data.batch)
        
        return mol_repr


    def forward(self, data1, data2):
        mol_repr1 = self.forward_pure(data1)
        mol_repr2 = self.forward_pure(data2)

        x = data1.molar_ratio.reshape(-1, 1).float()
 
        inter_repr = self.inter_module(mol_repr1, mol_repr2)
        inter_repr = torch.cat([inter_repr, x, 1-x], dim=1)
        energy_distribution = self.pred_boltzmann(inter_repr, data1.temps)
        delta_visc = self.pred_extra(energy_distribution).squeeze(-1)

        return delta_visc