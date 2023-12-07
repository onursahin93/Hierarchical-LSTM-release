"""
Created on
@author: onursahin93
"""


class DatasetFactory():


    def get_data_reader(self, agent_type, **kwargs):
        if agent_type == 'sine':
            from Data.DataReader.SineWave.sine_wave_reader import SineWaveReader
            return SineWaveReader(**kwargs)

        elif agent_type == 'linear':
            from Data.DataReader.LinearData.linear_data_reader import LinearDataReader
            return LinearDataReader(**kwargs)


