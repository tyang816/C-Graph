# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2021/12/02 16:26:12
@Author  :   Tan Yang 
@Version :   1.0
@Contact :   mashiroaugust@gmail.com
'''

# here put the import lib

from utils import data_tools as dt
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torchtext.legacy.data import Field, TabularDataset, Iterator
import math


class LocalEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_inputs, num_hiddens, num_layers):
        """
        extract features from the source code token sequence of the target function
        
        parms:
            num_inputs: GRU input size, default 128
            num_hiddens: GRU hidden size, default 128
            num_layers: GRU layer num, default 1
        """
        super(LocalEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.biGRU = nn.GRU(num_inputs, num_hiddens, num_layers, bidirectional=True)

    def forward(self, X):
        vec = self.embedding(X)
        output, state = self.biGRU(vec)
        return output, state


class classGraphDataset(InMemoryDataset):
    """
    build calss-graph dataset

    parms:
        root: data root directory
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(classGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.BASE_METHOD.json', 'data.CLASS_METHOD.json']

    @property
    def processed_file_names(self):
        return ['data.base.pt']

    def download(self):
        # download dataset from internet
        pass
            
    def class_graph(self, x, edge_index):
        data = Data(x=x, edge_index=edge_index)
        return data

    def process(self):
        base_path = self.raw_paths[0]
        class_path = self.raw_paths[1]
        method_field = torch.load('data/field/' + 'METHOD' + '_field.pkl')
        # [[class1_1, class1_2, ...,], [class2_1, class2_2, ...,]]
        node_class_list = dt.load(class_path, 'class') 
        # [base1, base2, ...,]
        node_base_list = dt.load(base_path, 'base')
        # each data in the list is a class level graph
        data_list = []
        # the index of `node_classes`([class1_1, class1_2, ...,]) corresponds to the index of `base`
        for node_classes in node_class_list:
            # create the node of target function
            node_list = [node_base_list[node_class_list.index(node_classes)]]
            # create the edges between classes and bases, default `[]`
            edge_index = [] 
            # iterate over the entire `node_classes` to create `edge_index` and graph
            for n_class in node_classes:
                if n_class not in node_list:
                    node_list.append(n_class)
                # every `n_class` corresponds to a `class` which is related to the `base_i`
                # where i means the index of `node_classes`, edges are bidirectional
                edge_index.append([0, node_list.index(n_class)])
                edge_index.append([node_list.index(n_class), 0])
            # convert to tensor
            x = method_field.process(node_list).T
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            data_list.append(self.class_graph(x, edge_index))
        # save the graph data
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])


class GlobalEncoder(nn.Module):
    def __init__(self, vocab_size, le_embed_size, le_num_inputs, le_num_hiddens, le_num_layers,
                 GAT_num_layer,GAT_dropout):
        """
        build C-Graph, vertex initialization and graph attention network

        parms:
            vocab_size: 
            le_embed_size: the embedding_size of the local endocer

        """
        super(GlobalEncoder, self).__init__()
        self.localEncoder = LocalEncoder(
            vocab_size, le_embed_size, le_num_inputs, le_num_hiddens, le_num_layers)
        self.GAT = GATConv(dropout=GAT_dropout)

        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # vertex initialization
        print(x.size())
        _, x = self.localEncoder(x)
        print(x)

        return 

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
    def forward(self, data):
        return

# X = torch.arange(25600, dtype=torch.int32).reshape((256, 100)).T
# print(X, X.size())
# localencoder = LocalEncoder(25600, 128, 128, 128, 1)
# out, hid = localencoder(X)
# print(out.size(), hid.size())

class_graph_data = classGraphDataset(root='./data') # 40 samples
data_loader = DataLoader(class_graph_data, batch_size=20, shuffle=True)
for batch in data_loader:
    globalencoder = GlobalEncoder(1704, 128, 128, 128, 1)
    print(globalencoder(batch))
    # print(batch)
    # print(batch.x)
    break

