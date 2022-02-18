# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/12/02 15:37:54
@Author  :   Tan Yang 
@Version :   1.0
@Contact :   mashiroaugust@gmail.com
'''

# here put the import lib

import torch
import yaml
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from model import ClassGraph
from data import classGraphDataset


def create_model(config):
    mdl = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda' if torch.cuda.is_available() else 'cpu')
    mdl = ClassGraph(config, device).to(device)
    print(torch.cuda.device_count())
    return mdl, device


# load config
config_path = './configs/small_config.yml'
config = yaml.load(open(config_path), Loader=yaml.FullLoader)

class_graph_data = classGraphDataset(root=DATA_HOME) # 40 samples
summary_field = torch.load(DATA_HOME + config['data']['field_summary'])
method_field = torch.load(DATA_HOME + config['data']['field_method'])
config['model']['com_vocab_size'] = len(summary_field.vocab)
config['model']['code_vocab_size'] = len(method_field.vocab)

print('com_vocab_size: {}'.format(config['model']['com_vocab_size']))
print('code_vocab_size: {}'.format(config['model']['code_vocab_size']))
print('batch_size: {}'.format(config['model']['batch_size']))
print('-' * 100)

# create model
net, device = create_model(config)
optimizer = optim.Adam(net.parameters(), lr=float(config['train']['lr']))
loss_fn = nn.CrossEntropyLoss()

# set up data generators


data_loader = DataLoader(class_graph_data, batch_size=config['model']['batch_size'], shuffle=True)

for batch in data_loader:
    # print('x.shape:',batch.x.shape)
    # print('x.shape:',batch.edge_index)
    classgraph(batch)

    
    history_acc = []
    history_loss = []
    for epoch in range(10):
        train_loss, valid_loss = [], []
        total_correct = 0.0
    