
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class LinearDataReader():
    def __init__(self,
                 data_dir="/home/onurs/PycharmProjects/SequentialPrediction/DataReader/LinearData/data",
                 L = 1000):
        """

        """
        super(LinearDataReader, self).__init__()

        # self.device = device
        self.data_dir = data_dir
        self.L = L
        self.next_value_prediction = True


    def read_data(self, data_dir=None):
        if data_dir is None:
            data_dir = self.data_dir
        try:
            x = np.load(f'{data_dir}/full_data.npy')
            x_missing = np.load(f'{data_dir}/data.npy', allow_pickle=True)
            mask = np.load(f'{data_dir}/mask.npy')
            t = np.load(f'{data_dir}/time.npy')
        except FileNotFoundError:
            print('The files could not be found. A new set is generating.')
            from Data.DataReader.LinearData.linear_data_generator import LinearDataGenerator
            generator = LinearDataGenerator(L=self.L)
            x, x_missing, mask, t = generator.generate_dataset()
        return x, x_missing, mask, t


if __name__ == '__main__':
    reader = LinearDataReader()
    x, x_missing, mask, t = reader.read_data()
    plt.plot(t, x, t, x_missing, 'ro')
    # plt.hold(True)
    # plt.plot(x_missing, 'ro')
    plt.show()

    print('done')
