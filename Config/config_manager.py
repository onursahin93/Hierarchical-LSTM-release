import argparse
from argparse import Namespace
from datetime import datetime
from Agent.Algorithm.algorithm_list import algorithm_list
from Data.dataset_list import dataset_list
from Config.default_config_factory import DefaultConfigFactory
from Config.constants import MAIN_DIR
from Config.utils.utils import unify_args
from Config.utils.utils import str2bool
from utils.utils import print_and_log
import os
from Config.basic_config import basic_config_dict as basic_config
from Config.logger_config import log_config_dict as log_config
from Config.data_config import data_config_dict as data_config
from Config.data_config import get_data_config
from Config.interaction_config import interaction_config_dict as interaction_config
from Config.save_load_config import save_load_dict as save_load_config
from Config.learning_config import learning_config_dict as learning_config


class ConfigManager():
    def __init__(self, optuna=False, optuna_dict=None, **kwargs):
        self.args = Namespace(**{"Paper": "HierarchicalLSTM"})
        parser = argparse.ArgumentParser(description='ConfigManager')

        parser.add_argument('--id',
                            type=bool,
                            default=True,
                            help='Whether it is a real run or not')

        parser.add_argument('--is-train',
                            type=bool,
                            default=True,
                            help='whether train or not')

        parser.add_argument('--is-demo',
                            type=bool,
                            default=False,
                            help='whether train or not')

        parser.add_argument('--algorithm',
                            type=str,
                            default="HierarchicalLSTM",
                            choices=algorithm_list,
                            help="agent's algorithm")

        parser.add_argument('--dataset',
                            type=str,
                            default="CaliforniaHousing",
                            choices=dataset_list,
                            help="Name of the Dataset")

        parser.add_argument('--use-default-alg-args',
                            type=bool,
                            default=True,
                            help='whether use default args or not')

        high_level_args = parser.parse_known_args()[0]

        if high_level_args.use_default_alg_args:
            default_args = DefaultConfigFactory().get_default_config(algorithm=high_level_args.algorithm)
            self.args = Namespace(**unify_args(vars(self.args), vars(high_level_args)))
            self.args = Namespace(**unify_args(vars(self.args), default_args))
        else:
            raise NotImplementedError
            self.args = Namespace(**unify_args(vars(self.args), vars(parser.parse_known_args()[0])))

        self.args = Namespace(**unify_args(vars(self.args), basic_config))
        self.args = Namespace(**unify_args(vars(self.args), save_load_config))
        self.args = Namespace(**unify_args(vars(self.args), log_config))
        # self.args = Namespace(**unify_args(vars(self.args), data_config))
        self.args = Namespace(**unify_args(vars(self.args), get_data_config(dataset=self.args.dataset)))
        self.args = Namespace(**unify_args(vars(self.args), interaction_config))
        self.args = Namespace(**unify_args(vars(self.args), learning_config))

        self.args.optuna = optuna
        if optuna:
            self.args = Namespace(**unify_args(vars(self.args), optuna_dict))


        # self.args = Namespace(**unify_args(vars(self.args), get_experiment_config(algorithm=self.args.algorithm)))

        self.__set_results_dir()
        self.__introduce()

    def __introduce(self):
        print(' ' * 26 + 'Options')
        for k, v in sorted(vars(self.args).items()):
            if not self.args.id:
                print(' ' * 26 + k + ': ' + str(v))
            else:
                print_and_log(self.args.params_file, k + ': ' + str(v), is_print=True)

        # print("debug")

    def __set_results_dir(self):
        if not self.args.id:
            self.args.results_dir = f'{MAIN_DIR}/results/{self.args.dataset}/{self.args.missing_ratio}/{self.args.optimizer}/{self.args.algorithm}{f"/window_length_{self.args.window_length}" if self.args.algorithm == "HierarchicalLSTM" else ""}/{self.args.id}'
        else:
            now = datetime.now()
            # self.args.results_dir = f'{MAIN_DIR}/results/{self.args.dataset}/normalize_{"True" if self.args.is_normalize else "False"}/seed_{self.args.seed}/{self.args.missing_ratio}/{self.args.optimizer}/{self.args.init}/{self.args.learning_rate}{f"/gamma_{self.args.gamma}" if self.args.is_decay else ""}/{self.args.algorithm}{f"/window_length_{self.args.window_length}" if self.args.algorithm == "HierarchicalLSTM" else ""}{f"/presence_pattern_{self.args.use_presence_patterns}" if self.args.algorithm == "HierarchicalLSTM" else ""}{f"/fc_tilde_different_{self.args.is_fc_tilde_different}" if self.args.algorithm == "HierarchicalLSTM" else ""}/{now.strftime("%Y%m%d-%H%M%S")}'
            self.args.results_dir = f'{MAIN_DIR}/results/' \
                                    f'{self.args.dataset}/' \
                                    f'{f"optuna/" if self.args.optuna else ""}' \
                                    f'normalize_{"True" if self.args.is_normalize else "False"}' \
                                    f'/seed_{self.args.seed}/{self.args.missing_ratio}/' \
                                    f'{self.args.optimizer}/' \
                                    f'{self.args.learning_rate}' \
                                    f'{f"/gamma_{self.args.gamma}" if self.args.is_decay else ""}/' \
                                    f'{self.args.algorithm}' \
                                    f'{f"/window_length_{self.args.window_length}" if self.args.algorithm == "HierarchicalLSTM" else ""}' \
                                    f'{f"/temp_{self.args.temperature}" if self.args.algorithm == "HierarchicalLSTM" else ""}/' \
                                    f'{now.strftime("%Y%m%d-%H%M%S")}'


        if not os.path.exists(self.args.results_dir):
            os.makedirs(self.args.results_dir)

        self.args.log_file = f'{self.args.results_dir}/{self.args.log_file}'
        self.args.params_file = f'{self.args.results_dir}/{self.args.params_file}'