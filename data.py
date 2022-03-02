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
from torch_geometric.data import InMemoryDataset,Data
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
config_path = './configs/large_config.yml'
config = yaml.load(open(config_path), Loader=yaml.FullLoader)
# data source
DATA_HOME = config['data']['home']
BASE_DATA = config['data']['base']
CLASS_DATA = config['data']['class']


class Vocab(object):
    def __init__(self, config):
        self.config = config
    
    def build_raw_data(self, data_name, category, key):
        """
        build the raw data
        """
        if category == 'base':
            assert isinstance(key, str), "get raw data of `base` need to declare the key word, like `method`"
            lines_base = dt.load_base(path=DATA_HOME+data_name, key=key, is_json=True)
            token_lines_base = dt.tokenize_code(lines_base)
            if key == 'method':
                dt.save(token_lines_base, DATA_HOME+self.config['data']['raw_base_method'], is_json=True)
            elif key == 'summary':
                dt.save(token_lines_base, DATA_HOME+self.config['data']['raw_base_summary'], is_json=True)
        elif category == 'class':
            lines_class_ = dt.load_class(path=DATA_HOME+data_name, key=key)
            if key == 'class_methods':
                key = 'method'
            token_lines_class_ = []
            for l in lines_class_:
                token_lines_class_.append(dt.tokenize_code(l))
            if key == 'method':
                dt.save(token_lines_class_, DATA_HOME+self.config['data']['raw_class_method'], is_json=True)
    
    def build_vocab(self, data_name, key):
        """
        build vocab from the raw data
        """
        if key == 'method':
            METHOD = Field(sequential=True, lower=True, init_token=bos, eos_token=eos, 
                           pad_token=pad, unk_token=unk, fix_length=self.config['model']['max_code_len'])
            # METHOD vocab can built by a list of files or a single file
            if isinstance(data_name, list):
                data = []
                for i in range(len(data_name)):
                    data = data + dt.load(DATA_HOME + data_name[i])
            elif isinstance(data_name, str):
                data = dt.load(DATA_HOME + data_name)
            METHOD.build_vocab(data)
            torch.save(METHOD, DATA_HOME + self.config['data']['field_method'])
        elif key == 'summary':
            SUMMARY = Field(sequential=True, lower=True, init_token=bos, eos_token=eos, 
                            pad_token=pad, unk_token=unk, fix_length=self.config['model']['max_com_len'])
            if isinstance(data_name, list):
                data = []
                for i in range(len(data_name)):
                    data = data + dt.load(DATA_HOME + data_name[i])
            elif isinstance(data_name, str):
                data = dt.load(DATA_HOME + data_name)
            SUMMARY.build_vocab(data)
            torch.save(SUMMARY, DATA_HOME + self.config['data']['field_summary'])
        else:
            return 
    
    def load_vocab(self, field_name):
        """
        load a field
        """
        return torch.load(DATA_HOME + field_name)
        

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
        return [config['data']['raw_base_method'].replace('/raw/',''), config['data']['raw_base_summary'].replace('/raw/',''),
         config['data']['raw_class_method'].replace('/raw/','')]

    @property
    def processed_file_names(self):
        return [config['data']['graph_dataset']]
            
    def class_graph(self, x, edge_index, y=None):
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    def process(self):
        base_method_path = self.raw_paths[0]
        base_summary_path = self.raw_paths[1]
        class_path = self.raw_paths[2]
        method_field = torch.load(DATA_HOME + config['data']['field_method'])
        summary_field = torch.load(DATA_HOME + config['data']['field_summary'])
        # [[method0_class0, method0_class1, ...,], [method1_class0, method1_class1, ...,]]
        class_list = dt.load(class_path, 'class') 
        # [method0, method1, ...,]
        base_method_list = dt.load(base_method_path, 'base')
        # [summary0, summary1, ...,]
        base_summary_list = dt.load(base_summary_path, 'base')
        # each data in the list is a class level graph
        data_list = []
        # the index of `classes`([method0_class0, method0_class1, ...,]) corresponds to the index of `base_method`
        for base_idx in range(len(base_method_list)):
            # create the node list of the target function
            node_list = [base_method_list[base_idx]]
            # create the summary
            summary = [base_summary_list[base_idx]]
            # create the edges between classes and bases, default `[]`
            edge_index = [] 
            # iterate over the entire `classes` to create `edge_index` and graph
            classes = class_list[base_idx]
            for c in classes:
                # max node number 100
                if len(node_list) >= 100:
                    break
                if c not in node_list:
                    node_list.append(c)
                # every `c` corresponds to a `class` which is related to the `base_i`
                # where i means the index of `classes`, edges are bidirectional
                edge_index.append([0, node_list.index(c)])
                edge_index.append([node_list.index(c), 0])
            for i in range(100 - len(node_list)):
                node_list.append(['<pad>'])
            # convert to tensor
            x = method_field.process(node_list).T
            y = summary_field.process(summary).T
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            data_list.append(self.class_graph(x, edge_index, y))
        # save the graph data
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])


if __name__ == '__main__':
    vocab = Vocab(config)
    print('-'*20 + 'Build raw data'+ '-'*20)
    vocab.build_raw_data(data_name=CLASS_DATA, category='class', key='class_methods')
    print('raw data of `class method` has been built, saved in {}'.format(DATA_HOME+config['data']['raw_class_method']))
    vocab.build_raw_data(data_name=BASE_DATA, category='base', key='method')
    print('raw data of `base method` has been built, saved in {}'.format(DATA_HOME+config['data']['raw_base_method']))
    vocab.build_raw_data(data_name=BASE_DATA, category='base', key='summary')
    print('raw data of `base method` has been built, saved in {}'.format(DATA_HOME+config['data']['raw_base_method']))
    
    print('-'*20 + 'Build summary and method vocab'+ '-'*20)
    vocab.build_vocab(data_name=[config['data']['raw_base_method'], config['data']['raw_class_method']], key='method')
    print('vocab of `method` has been built, saved in {}'.format(DATA_HOME+config['data']['field_method']))
    vocab.build_vocab(data_name=config['data']['raw_base_summary'], key='summary')
    print('vocab of `summary` has been built, saved in {}'.format(DATA_HOME+config['data']['field_summary']))
    
    print('-'*20 + 'Build class graph dataset'+ '-'*20)
    classGraphDataset(DATA_HOME, config)
    print('graph dataset saved in {}'.format(DATA_HOME+'/processed/'+config['data']['graph_dataset']))

    print('-'*20 + 'Test the `field`'+ '-'*20)
    method_vocab = vocab.load_vocab(config['data']['field_method'])
    summary_vocab = vocab.load_vocab(config['data']['field_summary'])
    method = [["override", "public", "object"]]
    summary = [["answers", "a", "copy", "of", "this", "object"]]
    print(method_vocab.process(method).T, method_vocab.process(method).T.size())
    print(summary_vocab.process(summary).T, summary_vocab.process(summary).T.size())

   