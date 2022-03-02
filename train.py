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
from model import ClassGraph, Net
from data import classGraphDataset
from tqdm import tqdm
import time

def create_model(config):
    mdl = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: cuda' if torch.cuda.is_available() else 'device: cpu')
    mdl = Net(config, device).to(device)
    print('cuda num: {}'.format(torch.cuda.device_count()))
    return mdl, device

def get_accuracy(scores, labels):
    correct = 0
    outputs = torch.argmax(scores, dim = 1)
    for predict, target in zip(outputs, labels):
        if predict == target:
            correct += 1
    return correct

def train(net, train_loader, optimizer, loss_fn, device):
    net.train()
    epoch_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # out: [com_len, batch_size, com_vocab_size]
        output = net(data)
        output_dim = output.shape[-1]
        # out: [(com_len - 1)*batch_size, com_vocab_size]
        output = output[1:].view(-1, output_dim)
        # label: [com_len, batch_size]
        label = data.y.transpose(1,0).to(device)
        # label: [(com_len - 1)*batch_size]
        label = label[1:].reshape(-1)
        loss = loss_fn(output, label)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
    return epoch_loss/len(train_loader)

def evaluate(net, valid_loader, loss_fn, device):
    net.eval()
    epoch_loss = 0
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            # out: [com_len, batch_size, com_vocab_size]
            output = net(data, False).detach()
            output_dim = output.shape[-1]
            # out: [(com_len - 1)*batch_size, com_vocab_size]
            output = output[1:].view(-1, output_dim)
            # label: [com_len, batch_size]
            label = data.y.transpose(1,0).to(device)
            # label: [(com_len - 1)*batch_size]
            label = label[1:].reshape(-1)
            loss = loss_fn(output, label)
            epoch_loss += loss.item()
    return epoch_loss/len(valid_loader)


def predict(net, test_loader, device):
    net.eval()
    preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # out: [com_len, batch_size, com_vocab_size]
            output = net(data, False).detach()
            # pre: [com_len, batch_size]
            pre = torch.zeros(output.shape[0], output.shape[1])
            for t in range(output.shape[0]):
                pre[t] = output[t].argmax(1)
            preds.append(pre)
    return preds

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def idx2com(ids, field):
    # ids: [batch_num, batch_size, len]
    seqs = []
    for seq_id in ids:
    # cur_ids: [batch_size, len]
        cur_ids = seq_id.transpose(1,0)
        for i in range(cur_ids.shape[0]):
            seq = []
            l = cur_ids[i].cpu().numpy().tolist()
        for j in l:
            seq.append(field.vocab.itos[int(j)])
        seqs.append(seq)
    return seqs

def idx2seq(ids, field):
    seqs = []
    for seq_id in ids:
        num = int(seq_id.shape[0] / 100)
        for i in range(num):
            seq = []
            l = seq_id[i].cpu().numpy().tolist()
        for j in l:
            seq.append(field.vocab.itos[int(j)])
        seqs.append(seq)
    return seqs

# load config
config_path = './configs/small_config.yml'
config = yaml.load(open(config_path), Loader=yaml.FullLoader)

# load data
class_graph_data = classGraphDataset(config['data']['home']) # 40 samples
class_graph_data = class_graph_data.shuffle()
graph_num = int(len(class_graph_data))
train_num = int(graph_num * 0.8)
valid_num = int(graph_num * 0.1)
train_data = class_graph_data[:train_num]
valid_data = class_graph_data[train_num:train_num+valid_num]
test_data = class_graph_data[train_num+valid_num:]
print('-' * 30 + 'DATA INFO' + '-' * 30)
print('graph_num: {}'.format(graph_num))
print('train_num: {}'.format(train_num))
print('valid_num: {}'.format(valid_num))
print('test_num: {}'.format(graph_num - train_num - valid_num))

# load field
summary_field = torch.load(config['data']['home'] + config['data']['field_summary'])
method_field = torch.load(config['data']['home'] + config['data']['field_method'])
config['model']['com_vocab_size'] = len(summary_field.vocab)
config['model']['code_vocab_size'] = len(method_field.vocab)
print('-' * 30 + 'FIELD INFO' + '-' * 30)
print('com_vocab_size: {}'.format(config['model']['com_vocab_size']))
print('code_vocab_size: {}'.format(config['model']['code_vocab_size']))
print('batch_size: {}'.format(config['model']['batch_size']))


# create model
net, device = create_model(config)
optimizer = optim.Adam(net.parameters(), lr=float(config['train']['lr']))
loss_fn = nn.CrossEntropyLoss(ignore_index=1).to(device)
train_loader = DataLoader(train_data, batch_size=config['model']['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=config['model']['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=config['model']['batch_size'], shuffle=True)

# train
history_acc = []
history_loss = []
history_valacc = []
hitory_valloss = []
print('-' * 30 + 'START TRAIN' + '-' * 30)

for epoch in range(1):
    start_time = time.time()
    
    train_loss = train(net, train_loader, optimizer, loss_fn, device)
    valid_loss = evaluate(net, valid_loader, loss_fn, device)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t Train Loss: {train_loss:.3f}')
    print(f'\t Valid Loss: {valid_loss:.3f}')
        



