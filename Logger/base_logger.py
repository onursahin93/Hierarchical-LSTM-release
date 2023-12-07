"""
Created on
@author: onursahin93
"""

import torch
import time
from utils.utils import replace_keys_values
from utils.utils import remove_new_lines


class BaseLogger():

    def __init__(self,
                 *args,
                 **kwargs):
        self.writer = None
        self.logging_dict = None

        self._create_logging_to_name()

    def __create_logging_list(self):
        from Config.LoggerConfig.logging_list import logging_list
        return logging_list


    def _create_logging_to_name(self):
        logging_list = self.__create_logging_list()
        self.logging_dict = {idx: logging_list[idx].split(";") for idx in
                             range(len(logging_list))}


    def log(self):
        raise NotImplementedError




# logger = BaseLogger()
#
# print('done')
#



