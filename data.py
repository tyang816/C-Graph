# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2021/12/02 16:07:45
@Author  :   Tan Yang 
@Version :   1.0
@Contact :   mashiroaugust@gmail.com
'''

# here put the import lib

import torch
import yaml
from collections import Counter, OrderedDict
# for `torchtext-0.11`, `Field` in the `torchtext.legacy`
from torchtext.legacy.data import Field, TabularDataset, Iterator
from torch_geometric.data import InMemoryDataset
import pandas as pd
# data prepocess tools
import utils.data_tools as dt


# define the reserved tokens
reserved_tokens = ['<unk>', '<pad>', '<s>', '</s>']
unk = '<unk>'
pad = '<pad>'
bos = '<s>'
eos = '</s>'


# load config
config_path = './configs/small_config.yml'
config = yaml.load(open(config_path), Loader=yaml.FullLoader)
# data source
DATA_HOME = config['data']['home']
BASE_DATA = config['data']['base']
CLASS_DATA = config['data']['class']
TOKEN_SIZE = config['preprocess']['token_size']


class Vocab(object):
    def __init__(self, root):
        self.root = root
    
    def build_raw_data(self, data_name, category, key):
        """
        build the raw data
        """
        if category == 'base':
            assert isinstance(key, str), "get raw data of `base` need to declare the key word, like `method`"
            lines_base = dt.load_base(path=self.root+data_name, key=key, is_json=True)
            token_lines_base = dt.tokenize_code(lines_base)
            dt.save(token_lines_base, '{}/raw/data.BASE_{}.json'.format(self.root, key.upper()), is_json=True)
        elif category == 'class':
            lines_class_ = dt.load_class(path=self.root+data_name, key=key)
            if key == 'class_methods':
                key = 'method'
            token_lines_class_ = []
            for l in lines_class_:
                token_lines_class_.append(dt.tokenize_code(l))
            dt.save(token_lines_class_, '{}/raw/data.CLASS_{}.json'.format(self.root, key.upper()), is_json=True)
    
    def build_vocab(self, data_name, key):
        """
        build vocab from the raw data
        """
        if key == 'method':
            METHOD = Field(sequential=True, lower=True, init_token=bos, eos_token=eos, 
                           pad_token=pad, unk_token=unk, fix_length=100)
            # METHOD vocab can built by a list of files or a single file
            if isinstance(data_name, list):
                data = []
                for i in range(len(data_name)):
                    data = data + dt.load(self.root + data_name[i])
            elif isinstance(data_name, str):
                data = dt.load(self.root + data_name)
            METHOD.build_vocab(data)
            torch.save(METHOD, '{}/field/field.METHOD.pkl'.format(self.root))
        elif key == 'summary':
            SUMMARY = Field(sequential=True, lower=True, init_token=bos, eos_token=eos, 
                            pad_token=pad, unk_token=unk)
            if isinstance(data_name, list):
                data = []
                for i in range(len(data_name)):
                    data = data + dt.load(self.root + data_name[i])
            elif isinstance(data_name, str):
                data = dt.load(self.root + data_name)
            SUMMARY.build_vocab(data)
            torch.save(SUMMARY, '{}/field/field.SUMMARY.pkl'.format(self.root))
        else:
            return 
    
    def load_vocab(self, field_name):
        """
        load a field
        """
        return torch.load(self.root + field_name)
        

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


if __name__ == '__main__':
    vocab = Vocab(DATA_HOME)
    vocab.build_raw_data(data_name=CLASS_DATA, category='class', key='class_methods')
    vocab.build_raw_data(data_name=BASE_DATA, category='base', key='method')
    vocab.build_raw_data(data_name=BASE_DATA, category='base', key='summary')

    vocab.build_vocab(data_name=['/raw/data.BASE_METHOD.json', '/raw/data.CLASS_METHOD.json'], key='method')
    vocab.build_vocab(data_name='/raw/data.BASE_SUMMARY.json', key='summary')

    # test the `field`
    method_vocab = vocab.load_vocab('/field/field.METHOD.pkl')
    summary_vocab = vocab.load_vocab('/field/field.SUMMARY.pkl')
    method = [["override", "public", "object"]]
    summary = [["answers", "a", "copy", "of", "this", "object"]]
    print(method_vocab.process(method).T, method_vocab.process(method).T.size())
    print(summary_vocab.process(summary).T, summary_vocab.process(summary).T.size())