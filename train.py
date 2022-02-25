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
from torch_geometric.loader import DataLoader
from model import ClassGraph
from data import classGraphDataset
from tqdm import tqdm


def create_model(config):
    mdl = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: cuda' if torch.cuda.is_available() else 'device: cpu')
    mdl = ClassGraph(config, device).to(device)
    print('cuda num: {}'.format(torch.cuda.device_count()))
    return mdl, device


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        print(unweighted_loss.shape)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss




loss = MaskedSoftmaxCELoss()
loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0]))

# load config
config_path = './configs/small_config.yml'
config = yaml.load(open(config_path), Loader=yaml.FullLoader)

# load data
class_graph_data = classGraphDataset(config['data']['home']) # 40 samples
class_graph_data = class_graph_data.shuffle()
print(class_graph_data[:1])
graph_num = int(len(class_graph_data))
train_num = int(graph_num * 0.8)
valid_num = int(graph_num * 0.1)
train_data = class_graph_data[:train_num]
valid_data = class_graph_data[train_num:train_num+valid_num]
test_data = class_graph_data[train_num+valid_num:]
print('-' * 30 + 'DATA_INFO' + '-' * 30)
print('graph_num: {}'.format(graph_num))
print('train_num: {}'.format(train_num))
print('valid_num: {}'.format(valid_num))
print('test_num: {}'.format(graph_num - train_num - valid_num))

# load field
summary_field = torch.load(config['data']['home'] + config['data']['field_summary'])
method_field = torch.load(config['data']['home'] + config['data']['field_method'])
config['model']['com_vocab_size'] = len(summary_field.vocab)
config['model']['code_vocab_size'] = len(method_field.vocab)
print('-' * 30 + 'FIELD_INFO' + '-' * 30)
print('com_vocab_size: {}'.format(config['model']['com_vocab_size']))
print('code_vocab_size: {}'.format(config['model']['code_vocab_size']))
print('batch_size: {}'.format(config['model']['batch_size']))


# create model
net, device = create_model(config)
optimizer = optim.Adam(net.parameters(), lr=float(config['train']['lr']))
loss_fn = nn.CrossEntropyLoss()
train_loader = DataLoader(train_data, batch_size=config['model']['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=config['model']['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=config['model']['batch_size'], shuffle=True)

# train
history_acc = []
history_loss = []
history_valacc = []
hitory_valloss = []
for epoch in tqdm(range(10)):
    net.train()
    train_loss, valid_loss = [], []
    total_correct = 0.0
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = net(data)
        label = data.y.to(device)
        loss = loss_fn(output.permute(0,2,1), label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    history_loss.append(loss_all)
    print(history_loss)
    net.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            pred = net(data).detach()


