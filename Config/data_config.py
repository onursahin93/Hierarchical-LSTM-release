from Config.utils.utils import unify_args

data_config_dict = {
    "train_ratio": 0.6,
    "missing_ratio": 0.1,
    "n_step": 1,
    "is_batch": False,
    "is_normalize": True}




import argparse
from Config.utils.utils import unify_known_args
from Config.utils.utils import str2bool

data_config_parser = argparse.ArgumentParser(description='LearningConfig')

data_config_parser.add_argument('--missing-ratio',
                                type=float,
                                default=None,
                                help='missing ratio')

data_config_parser.add_argument('--is-normalize',
                                type=str2bool,
                                default=None,
                                help='use normalization or not')



data_args = data_config_parser.parse_known_args()[0]
learning_config_dict = unify_known_args(data_config_dict, vars(data_args))






def get_data_config(dataset):
    if dataset == "NYSE":
        from Data.DataReader.NYSE.nyse_config import nyse_config as dataset_config
    elif dataset == "Kinematics":
        from Data.DataReader.Kinematics.kinematics_config import kinematics_config as dataset_config
    elif dataset == "CaliforniaHousing":
        from Data.DataReader.CaliforniaHousing.california_housing_config import california_housing_config as dataset_config
    elif dataset == "Pumadyn":
        from Data.DataReader.Pumadyn.pumadyn_config import pumadyn_config as dataset_config
    elif dataset == "BTC":
        from Data.DataReader.BTCFull1H.btc_full_1h_config import btc_full_1h_config as dataset_config
    elif dataset == "Bank8FM":
        from Data.DataReader.Bank8FM.bank8fm_config import bank8fm_config as dataset_config
    elif dataset == "Bank32NH":
        from Data.DataReader.Bank32NH.bank32nh_config import bank32nh_config as dataset_config
    elif dataset == "Elevators":
        from Data.DataReader.Elevators.elevators_config import elevators_config as dataset_config
    else:
        print(f"dataset is not implemented in data config")
        experiment_config = {}
        raise NotImplementedError

    return unify_args(data_config_dict, dataset_config)

