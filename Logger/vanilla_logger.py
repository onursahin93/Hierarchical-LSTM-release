"""
Created on
@author: onursahin93
"""

from Logger.base_logger import BaseLogger
from torch.utils.tensorboard import SummaryWriter
# from tensorboard.plugins.hparams import api as hp
import numpy as np

class VanillaLogger(BaseLogger):
    def __init__(self,
                 *args,
                 **kwargs):
        BaseLogger.__init__(self, *args, **kwargs)
        params = kwargs.get('params', None)
        self.writer = SummaryWriter(log_dir=params.results_dir)

    def log(self, label, value, step, main_tag="train"):
        self.writer.add_scalar(f'{main_tag}/{label}', value, step)

    def log_hparams(self, params, loss=None):
        raise NotImplementedError
        self.writer.add_hparams(hparam_dict={"optimizer": params.optimizer,
                                             "learning_rate":  params.learning_rate},
                                metric_dict={"hparam/loss": (0 if loss is None else loss)})

    def close(self):
        self.writer.close()

    def convert_game_wins_2_win_ratio(self, game_wins, game_count):
        raise NotImplementedError
        # return (game_count + game_wins)/(2*(game_count))

if __name__ == '__main__':
    import random
    logger = VanillaLogger()
    game_count= 0
    game_wins = 0
    for step in range(1000):
        if random.random()>0.95:
            game_wins += 1 if random.random() > 0.6 else -1
            logger.log(game_wins=game_wins, game_count=game_count, step=step)
            game_count += 1


    logger.close()
    print('done')

