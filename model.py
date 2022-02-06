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


" extract features from the source code token sequence of the target function "
class LocalEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_inputs, num_hiddens, num_layers):
        """
        params:
            vocab_size: the lenth of vocab
            embed_size: 128
            num_inputs: GRU H_in, num_inputs == embed_size, 128
            num_hiddens: GRU H_out, 128
            num_layers: the number of GRU layer, 1
        """
        super(LocalEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.biGRU = nn.GRU(num_inputs, num_hiddens, num_layers, bidirectional=True)

    def forward(self, X):
        """
        params:
            X: the code sequence, (batch_size, token_size)
        """
        # embedded: [token_size, num_nodes, embed_size]
        embedded = self.embedding(X).permute(1,0,2)
        # out: [token_size, num_nodes, num_direction * H_out]
        # h_n(state): [num_direction * num_layers, num_nodes, H_out]
        out, state = self.biGRU(embedded)
        out = torch.cat((out[:,:,:128], torch.flip(out[:,:,128:], [0])), 2)
        # concat the last hidden states, reduce dimension
        # concated_state: [num_nodes, 2 * H_out]
        concated_state = torch.cat((state[0], state[1]), 1)
        return out, concated_state

" build C-Graph, vertex initialization and graph attention network "
class GlobalEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, GRU_num_inputs, GRU_num_hiddens, GRU_num_layers,
                 GAT_num_layers, GAT_in_features, GAT_out_features, GAT_dropout):
        """
        params:
            LocalEncoder: vocab_size, embed_size, GRU_num_inputs, GRU_num_hiddens, GRU_num_layers
            GAT_num_layers: the number of GAT layer, 4
            GAT_in_features: num_direction * H_out, 2 * 128
            GAT_out_features: num_direction * H_out, 2 * 128
            GAT_dropout: dropout, 0.1
        """
        super(GlobalEncoder, self).__init__()
        self.localEncoder = LocalEncoder(
            vocab_size, embed_size, GRU_num_inputs, GRU_num_hiddens, GRU_num_layers)
        self.GAT = GATConv(GAT_in_features, GAT_out_features, dropout=GAT_dropout,add_self_loops=False)
        self.GATs = nn.ModuleList([self.GAT for _ in range(GAT_num_layers)])
        
        
    def forward(self, data):
        """
        params:
            data: data of a graph
            data.x: [num_nodes, token_size]
            data.edge_index: [2, num_edges]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # vertex initialization
        # g: [token_size, num_nodes, num_direction * H_out] 
        # q_n: [num_nodes, 2 * H_out]
        q, q_n = self.localEncoder(x)
        print('q:',q.shape)
        g = q_n
        print('q_n:', q_n.shape)
        for GAT in self.GATs:
            g = GAT(g, edge_index)
        print('g:', g.shape)
        return g, q_n, q


" leverage a graph attention mechanism "
class Attention(nn.Module):
    def __init__(self, num_inputs, num_hiddens, key):
        """
        params:
            num_inputs:
            num_hiddens: 
        """
        super(Attention, self).__init__()
        self.W_ga = nn.Linear(num_inputs, num_hiddens, bias=False)
        self.type = key

    def forward(self, h, g):
        """
        params:
            h: the hidden vector of decoder GRU, [time_step, batch_size, H_out]
            g: the last layer hidden vector of the GAT, [num_nodes, GAT_out_features]
        """
        assert self.type in ['graph','local'], "key should be `graph` or `local`"
        self.cg = []
        self.gamma = []
        for h_i in h:
            # print(h_i.shape,self.W_ga(g).shape)
            w = self.W_ga(g)
            gamma_ij = f.softmax(torch.mm(w, h_i.T), 0)
            self.gamma.append(gamma_ij)
            # print('gamma_ij:', gamma_ij.shape) 
            cg_i = torch.mm(gamma_ij.T, g)
            # print('cg_i:', cg_i.shape)
            self.cg.append(cg_i)
        # len(self.cg) == decoder time step
        # context: [time_step, 2 * num_hiddens]
        context = self.cg[0]
        for i in range(1, len(self.cg)):
            context = torch.cat((context, self.cg[i]), 0)
        print(self.type + '_context:', context.shape)
        
        if self.type == 'local':
            beta = self.gamma[0].T
            print(beta.shape)
            for i in range(1, len(self.gamma)):
                beta = torch.cat((beta, self.gamma[i].T), 0)
            print('beta:',beta.shape)
            return context, beta
        return context
        
" pointer mechanism "
class Pointer(nn.Module):
    def __init__(self, vocab_size, num_inputs, num_hiddens):
        """
        params:
            num_inputs: the 1th dimension of the `[h_i || c_i || cg_i]`
            num_hiddens: embed_size, 128
        """
        super(Pointer, self).__init__()
        self.W_v = nn.Linear(num_inputs, 100)
        self.w_h = nn.Linear(4 * num_hiddens, 1, bias=False)
        self.w_c = nn.Linear(2 * num_hiddens, 1, bias=False)
        self.w_y = nn.Linear(num_hiddens, 1, bias=False)
        self.leakRelu = nn.LeakyReLU(0.2)
        self.fc_out = nn.Linear(100, vocab_size)

    def forward(self, h, c, cg, beta, y):
        """
        params:
            h: [time_step, 4 * H_out]
            c: [time_step, 2 * H_out]
            cg: [time_step, 2 * H_out]
            beta: [time_step, token_size]
            y: [time_step, H_out]
        """
        concat = torch.cat((h, c, cg), 1)
        P_vocab = f.softmax(self.W_v(concat), 1)
        print('P_vocab:',P_vocab.shape)
        p_gen = self.leakRelu(self.w_h(h) + self.w_c(c) + self.w_y(y)).repeat(1,100)
        print('p_gen:',p_gen.shape)
        P_w = torch.mul(p_gen, P_vocab) + torch.mul((torch.ones_like(p_gen) - p_gen), beta)
        print('P_w:', P_w.shape)
        pre = self.fc_out(P_w)
        print('pre:', pre.shape)
        return pre

" adpot a GRU and use the concatenation of global representation g_t and local representation q_n "
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_inputs, num_hiddens):
        """
        params:
            vocab_size: the lenth of vocab
            embed_size:
            num_inputs:
            num_hiddens: 
        """
        super(Decoder, self).__init__()
        self.num_hiddens = 4 * num_hiddens
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.GRU = nn.GRU(num_inputs, self.num_hiddens, bidirectional=False)
        self.GraphAttention = Attention(2 * embed_size, self.num_hiddens, 'graph')
        self.LocalAttention = Attention(2 * embed_size, self.num_hiddens, 'local')
        self.Pointer = Pointer(1704, 2 * self.num_hiddens, embed_size)

    def forward(self, X, glo_enc_outputs, loc_enc_hiddens, loc_enc_outputs):
        """
        params:
            X: summary, [batch_size, time_step]
            glo_enc_outputs: [num_nodes, 2 * num_hiddens]
            loc_enc_hiddens: GRU output, [num_nodes, 2 * num_hiddens]
            loc_enc_outputs: hidden vector of each `t`, [time_step, num_nodes, 2 * num_hiddens]
        """
        state = torch.cat((glo_enc_outputs[0], loc_enc_hiddens[0]), 0).repeat(1,1,1)
        print('de_ini_state:',state.shape)
        # embedded: [time_step, batch_size, H_out]
        embedded = self.embedding(X).permute(1,0,2)
        print('sum_emb:', embedded.shape)
        # out: [time_step, batch_size, 4 * H_out]
        out, _ = self.GRU(embedded, state)
        print('de_out:', out.shape)
        cg = self.GraphAttention(out, glo_enc_outputs)
        c, beta = self.LocalAttention(out, loc_enc_outputs[:,0,:])
        print(cg.shape, c.shape)
        pre = self.Pointer(out.squeeze(1), c, cg, beta, embedded.squeeze(1))
        return pre



