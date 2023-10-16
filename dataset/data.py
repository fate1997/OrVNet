import pickle
from copy import deepcopy

import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

Mol = Chem.Mol

import os
import random

import pandas as pd
from tqdm import tqdm

from .featurizer import MoleculeFeaturizer


class OrvData(Data):
    def __init__(self, smiles):
        super().__init__()

        # Identification
        self.smiles: str = smiles
        
        # Viscosity-related information
        self.temps: torch.FloatTensor = None
        self.y: torch.FloatTensor = None

    def __repr__(self):
        return f'OrvData(SMILES={self.smiles}, num_atomse={self.num_nodes}, edge_index={self.edge_index.shape})'    
            
 
class OrvDataset(Dataset):
    def __init__(self, 
                 split: str='train',
                 raw_file_name: str='viscosity_pure.pickle',
                 additional_features: list=None,
                 replace: bool=False, 
                 save_folder: str='.\dataset\processed_files'):
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.data_list = []
        
        if os.path.exists(save_folder) is False:
            os.makedirs(save_folder, exist_ok=True)

        self.raw_file_path = '.\\dataset/raw_files\\' + raw_file_name 
        self.additional_features = additional_features
        self.processed_path = os.path.join(save_folder, 
                                           'OrVDataset_'+raw_file_name.split('.')[0]+'_'+split+'.pt')
        
        if replace or not os.path.exists(self.processed_path):
            self._process()
        else:
            self.data_list = torch.load(self.processed_path)
    
    def _process(self):
        
        with open(self.raw_file_path, 'rb') as f:
            raw_dataset = pickle.load(f)[self.split]

        for i, raw_data in enumerate(tqdm(raw_dataset)):
            mol = Chem.MolFromSmiles(raw_data['SMILES'])
            if mol is None:
                print(f"{raw_data['SMILES']} can not generate valid molecule")
                continue
            
            data = OrvData(smiles=raw_data['SMILES'])
            data.y = torch.tensor([raw_data['Viscosity']], dtype=torch.float32)
            data.temps = torch.tensor([raw_data['Temperature']], dtype=torch.float32)
            assert data.temps.size() == data.y.size()

            featurizer = MoleculeFeaturizer(additional_features=self.additional_features)
            feature_dict = featurizer(mol)
            
            # edge index and edges
            data.edge_index = feature_dict['edge_index']
            data.x = torch.tensor(feature_dict['x'], dtype=torch.float32)
            data.num_nodes = data.x.size(0)

            # pos, subgraph_index
            if 'pos' in self.additional_features:
                if feature_dict['pos'] is None:
                    print('Warning: No positions found for smiles {}'.format(data.smiles))
                    continue
                data.pos = torch.tensor(feature_dict['pos'], dtype=torch.float32)

            if 'subgraph_index' in self.additional_features:
                data.triple_index = torch.LongTensor(feature_dict['triple_index'])
                data.quadra_index = torch.LongTensor(feature_dict['quadra_index'])
            
            if 'mol_desc' in self.additional_features:
                data.mol_desc = torch.FloatTensor(feature_dict['mol_desc']).unsqueeze(0)

            self.data_list.append(data)
        
        torch.save(self.data_list, self.processed_path)
    
    def logarithm(self):
        for data in self.data_list:
            data.y = torch.log(data.y)

    def shuffle(self, seed):
        r = random.random
        random.seed(seed)
        random.shuffle(self.data_list, random=r)

    def __getitem__(self, index):
        return self.data_list[index]
    
    def __len__(self):
        return len(self.data_list)
    
    def __repr__(self):
        return f'OrvDataset(num_mols={self.__len__()})'
    
    def flatten(self):
        flatten_dataset = []
        for data in self.data_list:
            for i in range(data.temps.size(1)):
                flatten_data = deepcopy(data)
                flatten_data.temps = data.temps[0][i].unsqueeze(0).unsqueeze(0)
                flatten_data.y = data.y[0][i].unsqueeze(0).unsqueeze(0)
                flatten_dataset.append(flatten_data)
        self.data_list = flatten_dataset

        return self
    

class OrvMixData(Data):
    def __init__(self, smiles1, smiles2):
        super().__init__()
        self.smiles1 = smiles1
        self.smiles2 = smiles2

        self.x1 = None
        self.x2 = None
        self.edge_index1 = None
        self.edge_index2 = None
        self.num_nodes1 = None
        self.num_nodes2 = None
        self.molar_ratio = None
        self.temp = None
    
    def __repr__(self):
        return f'OrvData(SMILES={self.smiles1, self.smiles2})'


class OrvMixDataset(Dataset):
    def __init__(self, csv_path, additional_features, save_folder: str='.\dataset\processed_files', replace=False):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        assert 'smiles1' in self.df.columns
        assert 'smiles2' in self.df.columns
        assert 'molar_ratio' in self.df.columns
        assert 'temperature' in self.df.columns
        assert 'viscosity' in self.df.columns

        self.additional_features = additional_features
        name = csv_path.split('\\')[-1].split('.')[0] + '.pt'
        self.processed_path = os.path.join(save_folder, name)

        self.data_list1 = []
        self.data_list2 = []

        if replace or not os.path.exists(self.processed_path):
            self._process()
        else:
            self.data_list1, self.data_list2 = torch.load(self.processed_path)
     
    def _process(self):

        for i in tqdm(range(len(self.df))):
            row = self.df.iloc[i, :]
            mol1 = Chem.MolFromSmiles(row.smiles1)
            mol2 = Chem.MolFromSmiles(row.smiles2)

            if mol1 is None:
                print(f"{row.smiles1} can not generate valid molecule")
                continue
                
            if mol2 is None:
                print(f"{row.smiles2} can not generate valid molecule")
                continue

            data1 = OrvData(row.smiles1)
            data2 = OrvData(row.smiles2)
            data1.y = torch.log(torch.tensor([row.viscosity], dtype=torch.float32).unsqueeze(0))
            data2.y = torch.log(torch.tensor([row.viscosity], dtype=torch.float32).unsqueeze(0))
            data1.temps = torch.tensor([row.temperature], dtype=torch.float32).unsqueeze(0)
            data2.temps = torch.tensor([row.temperature], dtype=torch.float32).unsqueeze(0)

            featurizer = MoleculeFeaturizer(additional_features=self.additional_features)
            
            ##########################
            ## molecule component 1 ##
            ##########################
            feature_dict1 = featurizer(mol1)
            
            # edge index and edges
            data1.edge_index = feature_dict1['edge_index']
            data1.x = torch.tensor(feature_dict1['x'], dtype=torch.float32)

            # pos, subgraph_index
            if 'pos' in self.additional_features:
                if feature_dict1['pos'] is None:
                    print('Warning: No positions found for smiles {}'.format(data1.smiles))
                    continue
                data1.pos = torch.tensor(feature_dict1['pos'], dtype=torch.float32)

            if 'subgraph_index' in self.additional_features:
                data1.triple_index = torch.LongTensor(feature_dict1['triple_index'])
                data1.quadra_index = torch.LongTensor(feature_dict1['quadra_index'])
            
            if 'mol_desc' in self.additional_features:
                data1.mol_desc = torch.FloatTensor(feature_dict1['mol_desc']).unsqueeze(0)
            
            ##########################
            ## molecule component 2 ##
            ##########################
            feature_dict2 = featurizer(mol2)
            
            # edge index and edges
            data2.edge_index = feature_dict2['edge_index']
            data2.x = torch.tensor(feature_dict2['x'], dtype=torch.float32)

            # pos, subgraph_index
            if 'pos' in self.additional_features:
                if feature_dict2['pos'] is None:
                    print('Warning: No positions found for smiles {}'.format(data2.smiles))
                    continue
                data2.pos = torch.tensor(feature_dict2['pos'], dtype=torch.float32)

            if 'subgraph_index' in self.additional_features:
                data2.triple_index = torch.LongTensor(feature_dict2['triple_index'])
                data2.quadra_index = torch.LongTensor(feature_dict2['quadra_index'])
            
            if 'mol_desc' in self.additional_features:
                data2.mol_desc = torch.FloatTensor(feature_dict2['mol_desc']).unsqueeze(0)
            

            data1.num_nodes = data1.x.size(0)
            data1.molar_ratio = row.molar_ratio
            data2.num_nodes = data2.x.size(0)

            self.data_list1.append(data1)
            self.data_list2.append(data2)
        
        torch.save((self.data_list1, self.data_list2), self.processed_path)
    
    def __len__(self):
        assert len(self.data_list1) == len(self.data_list2)
        return len(self.data_list1)
    
    def __repr__(self):
        return f'OrvMixDataset(num_points: {len(self.data_list1)})'