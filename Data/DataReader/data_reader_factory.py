"""
Created on
@author: onursahin93
"""


class DataReaderFactory():

    def get_data_reader(self, params):
        dataset = params.dataset
        if dataset == 'NYSE':
            from Data.DataReader.NYSE.nyse_reader import NYSEDataReader
            return NYSEDataReader(params)

        elif dataset == 'Kinematics':
            from Data.DataReader.Kinematics.kinematics_reader import KinematicsDataReader
            return KinematicsDataReader(params)

        elif dataset == 'CaliforniaHousing':
            from Data.DataReader.CaliforniaHousing.california_housing_reader import CaliforniaHousingDataReader
            return CaliforniaHousingDataReader(params)
        elif dataset == "Pumadyn":
            from Data.DataReader.Pumadyn.pumadyn_reader import PumadynDataReader
            return PumadynDataReader(params)
        elif dataset == "BTC":
            from Data.DataReader.BTCFull1H.btc_full_1h_reader import BTCDataReader
            return BTCDataReader(params)
        elif dataset == "Bank8FM":
            from Data.DataReader.Bank8FM.bank8fm_reader import Bank8FMDataReader
            return Bank8FMDataReader(params)
        elif dataset == "Bank32NH":
            from Data.DataReader.Bank32NH.bank32nh_reader import Bank32NHDataReader
            return Bank32NHDataReader(params)
        elif dataset == "Elevators":
            from Data.DataReader.Elevators.elevators_reader import ElevatorsDataReader
            return ElevatorsDataReader(params)
        else:
            print(f"dataset is not implemented in data reader factory")
            raise NotImplementedError
