learning_config_dict = {"optimizer": "Adam",
                        "init": "normal_",
                        "normal_std": 0.01,
                        "init": "xavier_normal",
                        "learning_rate": 1e-4,
                        "is_decay": False,
                        "gamma": 1.0,
                        "is_y_always_exists": True}

initiation_list = ["normal",
                   "uniform",
                   "xavier_normal",
                   "xavier_uniform",
                   "default"
                   ]

import argparse
from Config.utils.utils import unify_known_args

learning_config_parser = argparse.ArgumentParser(description='LearningConfig')

learning_config_parser.add_argument('--learning-rate',
                                    type=float,
                                    default=None,
                                    help='Learning rate')

learning_config_parser.add_argument('--optimizer',
                                    type=str,
                                    default=None,
                                    help='optimizer')

learning_config_parser.add_argument('--rnn-hidden-size',
                                    type=int,
                                    default=None,
                                    help='overrides the default value of the algorithm')

learning_config_parser.add_argument('--init',
                                    type=str,
                                    default=None,
                                    choices=initiation_list,
                                    help='initiation type')

learning_config_parser.add_argument('--gamma',
                                    type=float,
                                    default=None,
                                    help='gamma')

learning_config_parser.add_argument('--normal-std',
                                    type=float,
                                    default=None,
                                    help='normal_std')

learning_args = learning_config_parser.parse_known_args()[0]
learning_config_dict = unify_known_args(learning_config_dict, vars(learning_args))
