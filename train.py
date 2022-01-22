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
from model import LocalEncoder, GlobalEncoder, Decoder

# load config
config_path = './configs/small_config.yml'
config = yaml.load(open(config_path), Loader=yaml.FullLoader)



# train config


# model config
TOKEN_SIZE = config['model']['token_size']
BIGRU_LAYER = config['model']['biGRU_layer']
BIGRU_NUM_HIDDEN = config['model']['num_hidden']

inputs = torch.zeros((5, TOKEN_SIZE), dtype=torch.long)
print(inputs.shape)
local_encoder = LocalEncoder(5,128,128, 128, BIGRU_LAYER)
output, state = local_encoder(inputs)
print(output.shape, state.shape)
