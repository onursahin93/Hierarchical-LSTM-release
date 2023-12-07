class DefaultConfigFactory():

    def get_default_config(self, algorithm):
        if algorithm == 'HierarchicalLSTM':
            from Config.AlgorithmConfig.HierarchicalLSTM.hierarchical_lstm_config import default_dict
            return default_dict
        else:
            raise NotImplementedError