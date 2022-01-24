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
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torchtext.legacy.data import Field, TabularDataset, Iterator
import math

def _fix_enc_hidden(hidden):
    return torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2).squeeze(0)

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
        vec = self.embedding(X).permute(1,0,2)
        _, state = self.biGRU(vec)
        # concat the last hidden states
        concated_state = _fix_enc_hidden(state)
        return concated_state


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
    def __init__(self, vocab_size, embed_size, GRU_num_inputs, GRU_num_hiddens, GRU_num_layers,
                 GAT_num_layers, GAT_in_features, GAT_out_features, GAT_dropout):
        """
        build C-Graph, vertex initialization and graph attention network

        parms:
            vocab_size: 
            le_embed_size: the embedding_size of the local endocer

        """
        super(GlobalEncoder, self).__init__()
        self.localEncoder = LocalEncoder(
            vocab_size, embed_size, GRU_num_inputs, GRU_num_hiddens, GRU_num_layers)
        self.GAT = GATConv(GAT_in_features, GAT_out_features, dropout=GAT_dropout,add_self_loops=False)
        self.GATs = nn.ModuleList([self.GAT for _ in range(GAT_num_layers)])
        
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # vertex initialization
        g = self.localEncoder(x)
        q_n = g[0]
        print(q_n.size())
        for GAT in self.GATs:
            g = GAT(g, edge_index)
        g_t = g[0]
        print(g_t.size())

        return g_t, q_n



class Decoder(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        """
        adpot a GRU and use the concatenation of global representation g_t and local representation q_n
        """
        super(Decoder, self).__init__()
        self.GRU = nn.GRU(num_inputs, num_hiddens, bidirectional=False)

    def init_state(self, glo_enc_outputs, loc_enc_outputs):
        return torch.cat((glo_enc_outputs.unsqueeze(0), loc_enc_outputs.unsqueeze(0)), 0)

    def graph_attention(self):
        return
    
    def local_attention(self):
        return 
    
    def pointer(self):
        return

    def forward(self, X, state):
        return 


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
print(class_graph_data[0])
globalencoder = GlobalEncoder(vocab_size=1704+1, embed_size=128, GRU_num_inputs=128, GRU_num_hiddens=128, GRU_num_layers=1,
                              GAT_num_layers=4, GAT_in_features=256, GAT_out_features=256, GAT_dropout=0.1)
    
glo_enc_outputs, loc_enc_outputs = globalencoder(class_graph_data[0])

decoder = Decoder(128, 128)
state = decoder.init_state(glo_enc_outputs, loc_enc_outputs)
print(state.size())
# data_loader = DataLoader(class_graph_data, batch_size=20, shuffle=True)
# for batch in data_loader:
#     print(batch.num_graphs)
#     # print(batch)
#     # print(batch.x)
#     break

