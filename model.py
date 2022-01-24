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
from data import classGraphDataset
import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.data import Data
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
        embedded = self.embedding(X).permute(1,0,2)
        _, state = self.biGRU(embedded)
        print('state', state.size())
        # concat the last hidden states
        concated_state = _fix_enc_hidden(state)
        print('concated_state', concated_state.size())
        return concated_state


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
    def __init__(self, vocab_size, embed_size, num_inputs, num_hiddens):
        """
        adpot a GRU and use the concatenation of global representation g_t and local representation q_n
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.GRU = nn.GRU(num_inputs, num_hiddens, bidirectional=False)

    def init_state(self, glo_enc_outputs, loc_enc_outputs):
        return torch.cat((glo_enc_outputs, loc_enc_outputs), 0).unsqueeze(0)

    def graph_attention(self):
        return
    
    def local_attention(self):
        return 
    
    def pointer(self):
        return

    def forward(self, X, state):
        embedded = self.embedding(X).permute(1,0,2)
        out, hid = self.GRU(embedded)
        print(out.size, hid.size())
        return 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
    def forward(self, data):
        return


class_graph_data = classGraphDataset(root='./data') # 40 samples
print(class_graph_data[0])
globalencoder = GlobalEncoder(vocab_size=1704+1, embed_size=128, GRU_num_inputs=128, GRU_num_hiddens=128, GRU_num_layers=1,
                              GAT_num_layers=4, GAT_in_features=256, GAT_out_features=256, GAT_dropout=0.1)

glo_enc_outputs, loc_enc_outputs = globalencoder(class_graph_data[0])

decoder = Decoder(vocab_size=1704+1, embed_size=128, num_inputs=128, num_hiddens=128)
state = decoder.init_state(glo_enc_outputs, loc_enc_outputs)
print(state.size())
# data_loader = DataLoader(class_graph_data, batch_size=20, shuffle=True)
# for batch in data_loader:
#     print(batch.num_graphs)
#     # print(batch)
#     # print(batch.x)
#     break

