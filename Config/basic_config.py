basic_config_dict = {"use_gpu": True,
                     "device": "cpu",
                     "torch_deterministic": True,
                     "seed": 1223334444}


import argparse
from Config.utils.utils import unify_known_args

basic_config_parser = argparse.ArgumentParser(description='BasicConfig')

basic_config_parser.add_argument('--device',
                                type=str,
                                default=None,
                                help='device')

basic_config_parser.add_argument('--seed',
                                type=int,
                                default=None,
                                help='seed')



basic_args = basic_config_parser.parse_known_args()[0]
basic_config_dict = unify_known_args(basic_config_dict, vars(basic_args))