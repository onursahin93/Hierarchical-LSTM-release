import numpy as np


def print_and_log(file, text, is_print=False):
    if is_print:
        print(text)
    print(text, file=open(f'{file}', 'a'))

def calculate_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def replace_keys_values(dictionary):
    return {dictionary[key] : key for key in dictionary.keys()}


def remove_new_lines(list_):
    for i, line in enumerate(list_):
        list_[i] = line.rstrip('\n')
    return list_

def list_idx_val_to_dict(list_):
    return {idx: list_[idx] for idx in range(len(list_))}

def get_element_list_dict(dictionary, key_list=None, idx=None):
    new_dict = {}
    if key_list is not None:
        for key in key_list:
            new_dict[key] = dictionary[key][idx]
    return new_dict
