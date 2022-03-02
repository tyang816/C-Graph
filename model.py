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


def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)
    return output

class LocalEncoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.batch_size = int(config['model']['batch_size'])
        self.embed_size = int(config['model']['embed_size'])
        self.hidden_size = int(config['model']['hidden_size'])
        self.code_vocab_size = int(config['model']['code_vocab_size'])

        self.CodeEmbed = nn.DataParallel(nn.Embedding(self.code_vocab_size, self.embed_size))
        self.biGRU = nn.DataParallel(nn.GRU(self.embed_size, self.hidden_size, 1, bidirectional=True, batch_first=True))
    
    def forward(self, x):
        # code_embed: [batch_size, code_len] => [batch_size, code_len, embed_size]
        code_embed = self.CodeEmbed(x)
        # out: [batch_size, code_len, hidden_size]
        out, state = self.biGRU(code_embed)
        # q: [batch_size, code_len, 2 * hidden_size]
        q = torch.cat((out[:, :, :128], torch.flip(out[:, :, 128:], [0])), 2)
        # concat the last hidden states
        # q_n: [batch_size * node_num, 2 * hidden_size]
        q_n = torch.cat((state[0], state[1]), 1)
        return q, q_n


class GlobalEncoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.gat_input_size = int(config['model']['gat_input_size'])
        self.gat_hidden_size = int(config['model']['gat_hidden_size'])
        self.gat_dropout = int(config['model']['gat_dropout'])

        self.GAT = nn.DataParallel(GATConv(self.gat_input_size, self.gat_hidden_size, dropout=self.gat_dropout))

    def forward(self, g, edge_index):
        for i in range(4):
            g = self.GAT(g, edge_index)
        return g


class Decoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.batch_size = int(config['model']['batch_size'])
        self.embed_size = int(config['model']['embed_size'])
        self.hidden_size = int(config['model']['hidden_size'])
        self.com_vocab_size = int(config['model']['com_vocab_size'])
        self.max_code_len = int(config['model']['max_code_len'])
        self.max_com_len = int(config['model']['max_com_len'])
        self.max_node_num = int(config['model']['max_node_num'])
        self.gat_input_size = int(config['model']['gat_input_size'])

        self.fc = nn.DataParallel(nn.Linear(2 * self.gat_input_size, self.hidden_size))
        self.ComEmbed = nn.DataParallel(nn.Embedding(self.com_vocab_size, self.embed_size))
        self.DeGRU = nn.DataParallel(nn.GRU(self.embed_size, self.hidden_size, 1, batch_first=True))
        self.W_ga = nn.DataParallel(nn.Linear(self.gat_input_size, self.hidden_size, bias=False))
        self.W_la = nn.DataParallel(nn.Linear(self.gat_input_size, self.hidden_size, bias=False))
        self.W_v = nn.DataParallel(nn.Linear(5 * self.hidden_size, self.max_code_len))
        self.w_h = nn.DataParallel(nn.Linear(self.hidden_size, 1, bias=False))
        self.w_c = nn.DataParallel(nn.Linear(2 * self.hidden_size, 1, bias=False))
        self.w_y = nn.DataParallel(nn.Linear(self.hidden_size, 1, bias=False))
        self.leakRelu = nn.DataParallel(nn.LeakyReLU(0.2))
        self.fc_out = nn.DataParallel(nn.Linear(self.max_code_len, self.com_vocab_size))

    def forward(self, y, q, q_n, g):
        # g_reshape: [batch_size, node_num, 2 * hidden_size]
        g_reshape = g.contiguous().view(self.batch_size, -1, 2 * self.hidden_size)
        # qn_reshape: [batch_size, node_num, 2 * hidden_size]
        qn_reshape = q_n.contiguous().view(self.batch_size, -1, 2 * self.hidden_size)
        # q_reshape: [batch_size, node_num, code_len, 2 * hidden_size]
        q_reshape = q.contiguous().view(self.batch_size, self.max_node_num, -1, 2 * self.hidden_size)
        # g_t: [batch_size, gat_input_size]
        g_t = g_reshape[:,0,:].squeeze(1)
        # qn_t: [batch_size, 2 * hidden_size]
        qn_t = qn_reshape[:,0,:].squeeze(1)
        # q_t: [batch_size, code_len, 2 * hidden_size]
        q_t = q_reshape[:,0,:,:].squeeze(1)
        
        # y: [batch_size] => [batch_size, 1]
        y = y.unsqueeze(1)
        # com_embed: [batch_size, 1] => [batch_size, 1, embed_size]
        com_embed = self.ComEmbed(y)
        # de_init_state: [batch_size, 2 * gat_input_size] => [1, batch_size, hidden_size]
        de_init_state = self.fc(torch.cat((qn_t, g_t), 1)).unsqueeze(0)
        
        # out: [batch_size, 1, hidden_size]
        out, state = self.DeGRU(com_embed, de_init_state)


        """Graph Attention"""
        # g_: [batch_size, node_num, hidden_size]
        g_ = self.W_ga(g_reshape).transpose(1,2)
        # [batch_size, 1, hidden_size] x [batch_size, hidden_size, node_num]
        # socres/gamma: [batch_size, 1, node_num]
        scores = torch.bmm(out, g_)
        gamma = f.softmax(scores, dim=-1)
        # [batch_size, 1, node_num] x [batch_size, node_num, 2 * hidden_size]
        # cg: [batch_size, 1, 2 * hidden_size]
        cg = torch.bmm(gamma, g_reshape)

        """Local Attention"""
        # q_: [batch_size, hidden_size, code_len]
        q_ = self.W_la(q_t).transpose(1,2)
        # [batch_size, 1, hidden_size] x [batch_size, hidden_size, code_len]
        # socres/beta: [batch_size, 1, code_len]
        scores = torch.bmm(out, q_)
        beta = f.softmax(scores, dim=-1)
        # [batch_size, 1, code_len] x [batch_size, code_len, 2 * hidden_size]
        # c: [batch_size, 1, 2 * hidden_size]
        c = torch.bmm(beta, q_t)
        
        """Pointer"""
        concat = torch.cat((out, c, cg), 2)
        # [batch_size, 1, 5 * hidden_size] => [batch_size, 1, code_len]
        P_vocab = f.softmax(self.W_v(concat), dim=1)
        # p_gen: [batch_size, 1]
        p_gen = self.leakRelu(self.w_h(out) + self.w_c(c) + self.w_y(com_embed)).squeeze(2)
        # [batch_size, 1, code_len] * [batch_size, 1]
        # P_w: [batch_size, 1, code_len]
        P_w = element_wise_mul(P_vocab, p_gen) + element_wise_mul(beta, torch.ones_like(p_gen) - p_gen)
        # pre: [batch_size, com_vocab_size]
        pre = self.fc_out(P_w).squeeze(1)
        return pre


class Net(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.batch_size = int(config['model']['batch_size'])
        self.com_vocab_size = int(config['model']['com_vocab_size'])
        self.localEncoder = LocalEncoder(config, device)
        self.globalEncoder = GlobalEncoder(config, device)
        self.decoder = Decoder(config, device)


    def forward(self, data, teacher_force=True):
        x, edge_index, y = data.x, data.edge_index, data.y
        # q: [batch_size, code_len, 2 * hidden_size]
        # q_n: [batch_size * node_num, 2 * hidden_size]
        q, q_n = self.localEncoder(x)
        # q: [batch_size * node_num, 2 * hidden_size]
        g = self.globalEncoder(q_n, edge_index)

        y_len = y.shape[1]
        outputs = torch.zeros(y_len, self.batch_size, self.com_vocab_size).to(self.device)
        # y: [com_len, batch_size]
        y = y.transpose(1,0)
        # init_put: [batch_size]
        init_put = y[0,:]
        if teacher_force:
            for t in range(1, y_len):
                output = self.decoder(init_put, q, q_n, g)
                outputs[t] = output
                init_put = y[t]
        else:
            for t in range(1, y_len):
                output = self.decoder(init_put, q, q_n, g)
                outputs[t] = output
                top1 = output.argmax(1).detach()
                init_put = top1
        # outputs: [com_len, batch_size, com_vocab_size]
        return outputs


class ClassGraph(nn.Module):
    def __init__(self, config, device):
        super(ClassGraph, self).__init__()
        self.name = 'class level graph'
        self.config = config
        self.batch_size = int(config['model']['batch_size'])
        self.embed_size = int(config['model']['embed_size'])
        self.hidden_size = int(config['model']['hidden_size'])
        self.code_vocab_size = int(config['model']['code_vocab_size'])
        self.com_vocab_size = int(config['model']['com_vocab_size'])
        self.max_code_len = int(config['model']['max_code_len'])
        self.max_com_len = int(config['model']['max_com_len'])
        self.max_node_num = int(config['model']['max_node_num'])
        self.gat_input_size = int(config['model']['gat_input_size'])
        self.gat_hidden_size = int(config['model']['gat_hidden_size'])
        self.gat_dropout = int(config['model']['gat_dropout'])

        self.CodeEmbed = nn.DataParallel(nn.Embedding(self.code_vocab_size, self.embed_size))
        self.biGRU = nn.DataParallel(nn.GRU(self.embed_size, self.hidden_size, 1, bidirectional=True, batch_first=True))
        self.GAT = nn.DataParallel(GATConv(self.gat_input_size, self.gat_hidden_size, dropout=self.gat_dropout))
        self.fc = nn.DataParallel(nn.Linear(2 * self.gat_input_size, self.hidden_size))
        self.ComEmbed = nn.DataParallel(nn.Embedding(self.com_vocab_size, self.embed_size))
        self.DeGRU = nn.DataParallel(nn.GRU(self.embed_size, self.hidden_size, 1, batch_first=True))
        self.W_ga = nn.DataParallel(nn.Linear(self.gat_input_size, self.hidden_size, bias=False))
        self.W_la = nn.DataParallel(nn.Linear(self.gat_input_size, self.hidden_size, bias=False))
        self.W_v = nn.DataParallel(nn.Linear(5 * self.hidden_size, self.max_code_len))
        self.w_h = nn.DataParallel(nn.Linear(self.hidden_size, 1, bias=False))
        self.w_c = nn.DataParallel(nn.Linear(2 * self.hidden_size, 1, bias=False))
        self.w_y = nn.DataParallel(nn.Linear(self.hidden_size, 1, bias=False))
        self.leakRelu = nn.DataParallel(nn.LeakyReLU(0.2))
        self.fc_out = nn.DataParallel(nn.Linear(self.max_code_len, self.com_vocab_size))

    def forward(self, data):
        x, edge_index, y = data.x, data.edge_index, data.y
        """Enocoder"""
        # code_embed: [batch_size, code_len] => [batch_size, code_len, embed_size]
        code_embed = self.CodeEmbed(x)
        # out: [batch_size, code_len, hidden_size]
        out, state = self.biGRU(code_embed)
        # q: [batch_size, code_len, 2 * hidden_size]
        q = torch.cat((out[:, :, :128], torch.flip(out[:, :, 128:], [0])), 2)
        # concat the last hidden states
        # q_n: [batch_size * node_num, 2 * hidden_size]
        q_n = torch.cat((state[0], state[1]), 1)
        g = q_n
        for i in range(4):
            g = self.GAT(g, edge_index)
        # g_reshape: [batch_size, node_num, 2 * hidden_size]
        g_reshape = g.contiguous().view(self.batch_size, -1, 2 * self.hidden_size)
        # qn_reshape: [batch_size, node_num, 2 * hidden_size]
        qn_reshape = q_n.contiguous().view(self.batch_size, -1, 2 * self.hidden_size)
        # q_reshape: [batch_size, node_num, code_len, 2 * hidden_size]
        q_reshape = q.contiguous().view(self.batch_size, self.max_node_num, -1, 2 * self.hidden_size)

        """"Decoder"""
        # find the target node
        # g_t: [batch_size, gat_input_size]
        g_t = g_reshape[:,0,:].squeeze(1)
        # qn_t: [batch_size, 2 * hidden_size]
        qn_t = qn_reshape[:,0,:].squeeze(1)
        # q_t: [batch_size, code_len, 2 * hidden_size]
        q_t = q_reshape[:,0,:,:].squeeze(1)
        # de_init_state: [batch_size, 2 * gat_input_size]
        de_init_state = self.fc(torch.cat((qn_t, g_t), 1))
        # com_embed: [batch_size, com_len] => [batch_size, com_len, embed_size]
        com_embed = self.ComEmbed(y)
        # out: [batch_size, com_len, hidden_size]
        out, state = self.DeGRU(com_embed, de_init_state.unsqueeze(0))

        """Graph Attention"""
        # g_reshape_: [batch_size, hidden_size, node_num]
        g_ = self.W_ga(g_reshape).transpose(1,2)
        # [batch_size, com_len, hidden_size] x [batch_size, hidden_size, node_num]
        # socres/gamma: [batch_size, com_len, node_num]
        scores = torch.bmm(out, g_)
        gamma = f.softmax(scores, dim=-1)
        # [batch_size, com_len, node_num] x [batch_size, node_num, 2 * hidden_size]
        # cg: [batch_size, com_len, 2 * hidden_size]
        cg = torch.bmm(gamma, g_reshape)

        """Local Attention"""
        # q_: [batch_size, hidden_size, code_len]
        q_ = self.W_la(q_t).transpose(1,2)
        # [batch_size, com_len, hidden_size] x [batch_size, hidden_size, code_len]
        # socres/beta: [batch_size, com_len, code_len]
        scores = torch.bmm(out, q_)
        beta = f.softmax(scores, dim=-1)
        # [batch_size, com_len, code_len] x [batch_size, code_len, 2 * hidden_size]
        # c: [batch_size, com_len, 2 * hidden_size]
        c = torch.bmm(beta, q_t)
        
        """Pointer"""
        concat = torch.cat((out, c, cg), 2)
        # [batch_size, com_len, 5 * hidden_size] => [batch_size, com_len, code_len]
        P_vocab = f.softmax(self.W_v(concat), dim=1)
        # p_gen: [batch_size, com_len]
        p_gen = self.leakRelu(self.w_h(out) + self.w_c(c) + self.w_y(com_embed)).squeeze(2)
        # [batch_size, com_len, code_len] * [batch_size, com_len]
        # P_w: [batch_size, com_len, code_len]
        P_w = element_wise_mul(P_vocab, p_gen) + element_wise_mul(beta, torch.ones_like(p_gen) - p_gen)
        # pre: [batch_size, com_len, com_vocab_size]
        pre = self.fc_out(P_w)
        return pre