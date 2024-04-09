default_dict = {
    "window_length": 3,
    "rnn_hidden_size": 36,
    "use_presence_patterns": True,
    "is_fc_tilde_different": True,
    "temperature": 10.0,
    "different_learning_rates_for_lstms": False,
    "is_peephole": False
}

import argparse
from Config.utils.utils import unify_known_args
from Config.utils.utils import str2bool

config_parser = argparse.ArgumentParser(description='HierarchicalLSTMConfig')


config_parser.add_argument('--window-length',
                            type=int,
                            default=None,
                            help='overrides the default window length of the algorithm')

config_parser.add_argument('--use-presence-patterns',
                            type=bool,
                            default=None,
                            help='overrides the default use_presence_patterns of the algorithm')

config_parser.add_argument('--is-fc-tilde-different',
                            type=str2bool,
                            default=None,
                            help='overrides the default is_fc_tilde_different of the algorithm')

config_parser.add_argument('--temperature',
                            type=float,
                            default=None,
                            help='overrides the default temperature of the algorithm')

config_parser.add_argument('--different-learning-rates-for-lstms',
                            type=str2bool,
                            default=None,
                            help='overrides the default different_learning_rates_for_lstms of the algorithm')

config_parser.add_argument('--is-peephole',
                            type=str2bool,
                            default=None,
                            help='use peephole connections or not')




config_args = config_parser.parse_known_args()[0]
default_dict = unify_known_args(default_dict, vars(config_args))