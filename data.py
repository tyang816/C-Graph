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


SUMMARY = Field(sequential=True, lower=True, init_token=bos,
                eos_token=eos, pad_token=pad, unk_token=unk)
# fields = {'method': ('method', METHOD), 'summary': ('summary', SUMMARY)}



if __name__ == '__main__':
    # preprocess the data
    # for name in ['method','summary','signature']:
    #     if name in ['method', 'signature']:
    #         lines = dt.load(DATA_HOME, TRAIN_DATA, key=name, is_json=True)
    #         data = dt.tokenize_code(lines)

    #     elif name in ['summary']:
    #         lines = dt.load(DATA_HOME, TRAIN_DATA, key=name, is_json=True)

    #     train, val, test = data[],data[],data[]

    lines_base = dt.load_base(DATA_HOME+ BASE_DATA, key='method', is_json=True)
    lines_class = dt.load_class(DATA_HOME+ CLASS_DATA, is_vocab=True)
    lines_class_ = dt.load_class(DATA_HOME+ CLASS_DATA)
    token_lines_class_ = []
    for l in lines_class_:
        token_lines_class_.append(dt.tokenize_code(l))
    lines = lines_base + lines_class
    print(lines[:2])
    data = dt.tokenize_code(lines)

    # METHOD = Field(sequential=True, lower=True, init_token=bos,
    #                eos_token=eos, pad_token=pad, unk_token=unk, fix_length=100)
    # METHOD.build_vocab(data)


    # # save the vocab model and processed data
    # torch.save(METHOD, 'data/field/' + 'METHOD' + '_field.pkl')
    # dt.save(dt.tokenize_code(lines_base), 'data/raw/data.' + 'BASE_METHOD' + '.json', is_json=True)
    # dt.save(token_lines_class_, 'data/raw/data.' + 'CLASS_METHOD' + '.json', is_json=True)
    

    # test the `field`
    m2 = torch.load('data/field/' + 'METHOD' + '_field.pkl')
    print(len(m2.vocab))
    base = dt.tokenize_code(lines_base)
    print(m2.process(base).T,m2.process(base).T.size())