import torch
from torch_geometric.loader import DataLoader

from models.OrVNet_FFiNet import OrVNetModel
from utils import evaluate_score, load_config, save_results
from dataset.process import process_pure_compound
from typing import List


def predict_pure_compound_viscosity(smiles: str, 
                                    temperature: List[float]) -> List[float]:
    r"""Predict viscosity of a pure compound at a given temperature."""
    
    # Load config
    config_path = './configs/train.yml'
    config = load_config(config_path)
    
    # Load model
    model = OrVNetModel(config)
    model.load_state_dict(torch.load('./trained_models/OrVNet.pt'))
    model.eval()
    
    # Process input data
    data = process_pure_compound(smiles, 
                                 temperature, 
                                 config.data.additional_features)
    
    # Run model
    output = model(data).exp().tolist()[0]
    
    return output