import torch
from rdkit import Chem

from .data import OrvData, OrvMixData
from .featurizer import MoleculeFeaturizer


def process_pure_compound(smiles: str, 
                          temperature: float, 
                          additional_features: list=None):
    r"""Process a single compound at a given temperature."""
   
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"{smiles} can not generate valid molecule")
        return
    
    data = OrvData(smiles=smiles)
    data.temps = torch.tensor([temperature], dtype=torch.float32)

    featurizer = MoleculeFeaturizer(additional_features=additional_features)
    feature_dict = featurizer(mol)
    
    # edge index and edges
    data.edge_index = feature_dict['edge_index']
    data.x = torch.tensor(feature_dict['x'], dtype=torch.float32)
    data.num_nodes = data.x.size(0)

    # pos, subgraph_index
    if 'pos' in additional_features:
        if feature_dict['pos'] is None:
            print(f'Warning: No positions found for smiles {smiles}')
            return 
        data.pos = torch.tensor(feature_dict['pos'], dtype=torch.float32)

    if 'subgraph_index' in additional_features:
        data.triple_index = torch.LongTensor(feature_dict['triple_index'])
        data.quadra_index = torch.LongTensor(feature_dict['quadra_index'])
    
    if 'mol_desc' in additional_features:
        data.mol_desc = torch.FloatTensor(feature_dict['mol_desc']).unsqueeze(0)
    
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    
    return data