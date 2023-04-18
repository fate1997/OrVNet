from torch import nn
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import argparse
import os

from utils import EarlyStopping, evaluate_score, \
    save_results, load_config, get_logger, data2iter

from models.OrVNet_FFiNet import OrVNetModel
from models.FFiNet_concat import FFiNetConcatModel
from models.FFiNet_equation import FFiNetEquationModel


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config, logger):

    seed = config.train.seed
    save_folder = config.train.save_folder
    desc = config.desc
    model_type = config.model_type
    

    logger.info(f'############{desc}############')
    logger.info(vars(config))
    
    # Load train config
    num_epochs = config.train.num_epochs
    lr = config.train.lr 
    patience = config.train.patience
    device = config.train.device
    log_interval = config.train.log_interval

    # Load data
    logger.info("-----------Dataset Loading-----------")
    batch_size = config.data.batch_size
    name = config.data.name
    flatten = config.data.flatten
    logarithm = config.data.logarithm
    additional_features = config.data.additional_features
    replace = config.data.replace
    train_loader, val_loader, test_loader = data2iter(name, batch_size, additional_features, replace, flatten, logarithm)

    # training init
    seed_all(seed)
    early_stopping = EarlyStopping(patience=patience, 
                                   path=os.path.join(save_folder, 'checkpoints/'),
                                   model_type=model_type)
    
    criterion = nn.L1Loss()
    
    logger.info("------------Model Creating-----------")

    if config.model.name == 'OrVNet':
        model = OrVNetModel(config=config).to(device)
    elif config.model.name == 'FFiNetConcat':
        model = FFiNetConcatModel(config=config).to(device)
    elif config.model.name == 'FFiNetEquation':
        model = FFiNetEquationModel(config=config).to(device)
    else:
        raise NotImplementedError(f'Model Not implemented')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    logger.info("------------Train Running------------")
    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        num_examples = 0
        for i, batch in enumerate(train_loader):

            # forward
            model.to(device)
            batch = batch.to(device)

            y = batch.y
            outputs = model(batch)

            loss = criterion(outputs, y) # , batch.temps

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_examples += y.shape[0]
            loss_sum += loss.item() * y.shape[0]

        val_metric = evaluate_score(model, val_loader)

        if epoch % log_interval == 0:
            logger.info(f'epoch:{epoch}, loss = {loss_sum / num_examples: .4f}, '
                f'val loss = {val_metric["loss"]:.4f}, '
                f'val AARD = {np.round(val_metric["AARD"], decimals=4)}, '
                f'val R2 = {np.round(val_metric["R2"], decimals=4)}')
    
        # early stopping
        min_metrics = np.array(val_metric["AARD"]).mean()

        early_stopping(min_metrics, model)

        if early_stopping.early_stop:
            logger.info('------------Early stopping------------')
            break

    model.load_state_dict(torch.load(os.path.join(save_folder, 'checkpoints', model_type + '.pt')))
    
    test_metric = evaluate_score(model, test_loader)
    val_metric = evaluate_score(model, val_loader)

    logger.info(f'val AARD = {np.round(val_metric["AARD"], decimals=4)}, '
                f'val R2 = {np.round(val_metric["R2"], decimals=4)}, '
        f'test AARD = {np.round(test_metric["AARD"], decimals=4)}, '
        f'test R2 = {np.round(test_metric["R2"], decimals=4)}, ')

    save_results(model, train_loader, os.path.join(save_folder, 'pred_results'), model_type+'_train')
    save_results(model, val_loader, os.path.join(save_folder, 'pred_results'), model_type+'_val')
    save_results(model, test_loader, os.path.join(save_folder, 'pred_results'), model_type+'_test')
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-desc', default='Testing')
    parser.add_argument('-config_name', default='train.yml')
    parser.add_argument('-model', default='OrVNet')
    args = parser.parse_args()

    # load config
    config_path = os.path.join('.\configs', args.config_name)
    config = load_config(config_path)

    config.desc = args.desc
    config.model.name = args.model

    # logger
    save_folder = config.train.save_folder
    model_type = config.model.name + '_' + config.desc
    config.model_type = model_type
    logger = get_logger(model_type+'.log', os.path.join(save_folder, 'logs/'))
    
    train(config, logger)