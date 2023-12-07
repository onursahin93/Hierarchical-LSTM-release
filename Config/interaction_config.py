interaction_config_dict = {"num_epoch": 100,
                           "evaluation_interval": 1, #evaluate at each n episodes
                           "batch_size": 1
                           }


import argparse
from Config.utils.utils import unify_known_args

interaction_config_parser = argparse.ArgumentParser(description='InteractionConfig')


interaction_config_parser.add_argument('--num-epoch',
                                    type=int,
                                    default=None,
                                    help='Number of epoch')



interaction_args = interaction_config_parser.parse_known_args()[0]
interaction_config_dict = unify_known_args(interaction_config_dict, vars(interaction_args))
