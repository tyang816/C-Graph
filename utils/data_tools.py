# -*- encoding: utf-8 -*-
'''
@File    :   data_tools.py
@Time    :   2021/12/02 21:13:17
@Author  :   Tan Yang 
@Version :   1.0
@Contact :   mashiroaugust@gmail.com
'''

# here put the import lib

import json
import re
import torch

def load(path, key=None):
    """
    load raw data

    parms:
        path: data path
        key: category of the raw data
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    if key == None or key == 'base':
        for line in lines:
            l = eval(line)
            new_lines.append(l)
    elif key == 'class':
        for line in lines:
            l = eval(line)
            c_line = []
            for c in l:
                c_line.append(c)
            new_lines.append(c_line)
    return new_lines
        


def load_base(path, is_json=False, key=None, drop_list=()):
    """
    load base(target function) code
    
    parms:
        path: base.json path
        is_json: 
        key: load the value through a certain key
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not is_json:
        if not drop_list:
            return lines
        else:
            return [line for i, line in enumerate(lines) if not i in drop_list]
    
    if key is None:
        return [json.loads(line) for i, line in enumerate(lines) if not i in drop_list]
    else:
        return [json.loads(line)[key] for i, line in enumerate(lines) if not i in drop_list]


def load_class(path, key=None, source_id=None, is_vocab=False):
    """
    load class(function related to target function) code

    parms:
        path: class.json path
        source_id: load related class of `source_id`
        is_vocab: Load all the classes to append to the target function to make the vocab 
    """
    target_line = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # load target related class
    if source_id is not None:
        for line in lines:
            if json.loads(line)['id'] == source_id:
                target_line = json.loads(line)
                break
        related_class_list = target_line[key]
        related_class_full = []
        for c in related_class_list:
            related_class_full.append(str(c['full']))
        return related_class_full
    # load all
    else:
        related_class_list = [json.loads(line)[key] for i,line in enumerate(lines)]
        related_class_full = []
        # build vocab and return a one layer class list
        if is_vocab:
            for method in related_class_list:
                for code in method:
                    related_class_full.append(str(code['full']))
        # return a two layer class list
        else:
            for method in related_class_list:
                related_codes = []
                for code in method:
                    related_codes.append(str(code['full']))
                related_class_full.append(related_codes)
        return related_class_full

    

def tokenize_code(lines):
    """
    convert a string list(eg. ['a string', 'a word',...]) to token list(eg. [['a', 'string'], ['a', 'word'],...])

    parms:
        lines: a list of target tokens
    """
    new_lines = []
    for line in lines:
        # camelCase to undersocre
        line = re.sub(r'([a-z])([A-Z])',r'\1_\2', line)
        # underscore or none-alphabetical letters to space
        new_lines.append(str((re.sub(r'[^A-Za-z]+', ' ', line).strip().lower())).split())
    return new_lines


def save(data, path, is_json=False):
    """
    save the tokenized data (raw data)

    parms:
        data: the tokenized data
        path: data path
        is_json: 
    """
    print('Saving {}...'.format(str(path)))
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            if is_json:
                line = '' if not line else json.dumps(line)
            f.write(line + '\n')
