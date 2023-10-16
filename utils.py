import torch
from torch import nn
from torch_geometric.loader import DataLoader

import pandas as pd
import numpy as np
import os
import yaml
import logging
import logging.config
import json
import sys
from easydict import EasyDict
import random

from sklearn import metrics

from dataset.data import OrvDataset, OrvMixDataset


class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""

    def __init__(self, patience=100, verbose=False,
                 path='../results/trained_models/', model_type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.model_type = model_type

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model) 
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # Saves model when validation score doesn't improve in patience
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, self.model_type + '.pt'))
        self.val_loss_min = val_loss


def evaluate_score(model, data_loader, logarithm=True, train_mixture=False):
    if isinstance(model, nn.Module):
        model.eval()
    
    if train_mixture:
        y_true, y_pred, _, _, _ = mixture_batch_flatten(model, data_loader, logarithm)
    else:
        y_true, y_pred, _, _ = batch_flatten(model, data_loader, logarithm)


    metric_dict = {}
    metric_dict['AARD'] = np.abs(((np.array(y_true) - np.array(y_pred)) / np.array(y_true))).mean(0).tolist()
    metric_dict['R2'] = metrics.r2_score(y_true, y_pred)
    metric_dict['loss'] = nn.L1Loss()(torch.tensor(y_true, dtype=torch.float32), 
                                    torch.tensor(y_pred, dtype=torch.float32)).tolist()

    return metric_dict

def get_logger(name, log_dir, config_dir='./configs/'):

    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger 


def data2iter(split_type, batch_size, additional_features, replace, flatten, logarithm):
    
    # load dataset and split to train and test set
    train_dataset = OrvDataset('train', split_type, additional_features, replace)
    val_dataset = OrvDataset('val', split_type, additional_features, replace)
    test_dataset = OrvDataset('test', split_type, additional_features, replace)
    
    if flatten:
        train_dataset.flatten()
        val_dataset.flatten()
        test_dataset.flatten()
    
    if logarithm:
        train_dataset.logarithm()
        val_dataset.logarithm()
        test_dataset.logarithm()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def mixture2iter(csv_path, batch_size, train_ratio, additional_features, replace):
    
    dataset = OrvMixDataset(csv_path, additional_features, replace=replace)

    r = random.random
    random.seed(42)
    num_points = len(dataset)
    ids = list(range(num_points))
    random.shuffle(ids, random=r)

    split_1 = int(train_ratio * num_points)
    split_2 = int((train_ratio+(1-train_ratio)/2) * num_points)
    train_ids = ids[: split_1]
    val_ids = ids[split_1:split_2]
    test_ids = ids[split_2:]

    train_data1 = [dataset.data_list1[i] for i in train_ids]
    val_data1 = [dataset.data_list1[i] for i in val_ids]
    test_data1 = [dataset.data_list1[i] for i in test_ids]

    train_data2 = [dataset.data_list2[i] for i in train_ids]
    val_data2 = [dataset.data_list2[i] for i in val_ids]
    test_data2 = [dataset.data_list2[i] for i in test_ids]

    train_loader1 = DataLoader(train_data1, batch_size=batch_size)
    val_loader1 = DataLoader(val_data1, batch_size=batch_size)
    test_loader1 = DataLoader(test_data1, batch_size=batch_size)

    train_loader2 = DataLoader(train_data2, batch_size=batch_size)
    val_loader2 = DataLoader(val_data2, batch_size=batch_size)
    test_loader2 = DataLoader(test_data2, batch_size=batch_size)

    return (train_loader1, train_loader2), (val_loader1, val_loader2), (test_loader1, test_loader2)


# flatten batch
def batch_flatten(model, data_loader, logarithm=True):
    model.eval()
    model.to('cpu')
    y_true = []
    y_predict = []
    smiles_list = []
    temp_list = []
    for batch in data_loader:
        batch = batch.to('cpu')
        y_hat = model(batch).detach().flatten().tolist()
        y_true += batch.y.flatten().tolist()
        temp_list += batch.temps.flatten().tolist()
        y_predict += y_hat
        smiles_list += [smiles for smiles in batch.smiles for i in range(batch.temps.size(1))]
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)   
    
    if logarithm:
        y_true = np.exp(y_true).tolist()
        y_predict = np.exp(y_predict).tolist()
    return y_true, y_predict, smiles_list, temp_list


def mixture_batch_flatten(model, data_loader, logarithm=True):
    model.eval()
    model.to('cpu')
    y_true = []
    y_predict = []
    smiles_list1 = []
    smiles_list2 = []
    temp_list = []
    for batch1, batch2 in zip(data_loader[0], data_loader[1]):
        batch1 = batch1.to('cpu')
        batch2 = batch2.to('cpu')
        y_hat = model(batch1, batch2).detach().flatten().tolist()
        y_true += batch1.y.flatten().tolist()
        temp_list += batch1.temps.flatten().tolist()
        y_predict += y_hat
        smiles_list1 += [smiles for smiles in batch1.smiles for i in range(batch1.temps.size(1))]
        smiles_list2 += [smiles for smiles in batch2.smiles for i in range(batch1.temps.size(1))]
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)   
    
    if logarithm:
        y_true = np.exp(y_true).tolist()
        y_predict = np.exp(y_predict).tolist()
    return y_true, y_predict, smiles_list1, smiles_list2, temp_list


def save_results(model, data_loader, save_folder, model_type, train_mixture=False):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    
    if train_mixture:
        y_true, y_pred, smiles1, smiles2, temp_list = mixture_batch_flatten(model, data_loader)
        results = pd.DataFrame(np.array([smiles2, temp_list, y_true, y_pred]).T, columns=['smiles2', 'T', 'y_true', 'y_pred'], index=smiles1)
    else:
        y_true, y_pred, smiles, temp_list = batch_flatten(model, data_loader)
        results = pd.DataFrame(np.array([temp_list, y_true, y_pred]).T, columns=['T', 'y_true', 'y_pred'], index=smiles)

    results.to_csv(os.path.join(save_folder, model_type + '.csv'))


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))