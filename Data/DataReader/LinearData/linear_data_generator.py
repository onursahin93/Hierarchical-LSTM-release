import numpy as np
import numpy.ma as ma

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import inspect




class LinearDataGenerator():
    def __init__(self,
                 data_dir="/home/onurs/PycharmProjects/SequentialPrediction/DataReader/LinearData/data",
                 delta_t = 0.1,
                 f=10,
                 L=1000,
                 safe_margin =3,
                 missing_ratio = 0.2,
                 invert_mask=True,
                 save= False):
        """

        """
        super(LinearDataGenerator, self).__init__()
        self.L = L
        self.delta_t = delta_t
        self.f = f
        self.t = t = np.linspace(0.0, 5.0, num=L)
        self.safe_margin = safe_margin
        self.save = save
        self.missing_ratio = missing_ratio
        self.data_dir = data_dir
        self.invert_mask = invert_mask


    def generate_dataset(self,
                         delta_t = None,
                         f=None,
                         L=None,
                         missing_ratio = None,
                         save = None,
                         safe_margin=None,
                         data_dir= None):

        if delta_t is None:
            delta_t = self.delta_t
        if f is None:
            f = self.f
        if L is None:
            L = self.L
        if missing_ratio is None:
            missing_ratio = self.missing_ratio
        if save is None:
            save = self.save
        if data_dir is None:
            data_dir = self.data_dir
        if safe_margin is None:
            safe_margin = self.safe_margin

        t = np.arange(L)
        x = np.arange(L)
        is_missing_mask = np.random.rand(L) < missing_ratio
        is_missing_mask[:safe_margin] = False
        is_missing_mask[-safe_margin:] = False

        x_missing = ma.masked_array(data=x,
                                 mask=is_missing_mask,
                                 fill_value=None)

        if self.invert_mask:
            is_missing_mask = np.invert(is_missing_mask)

        if save:
            np.save(f'{data_dir}/full_data.npy', x)
            np.save(f'{data_dir}/mask.npy', is_missing_mask)
            x_missing.dump(f'{data_dir}/data.npy')
            np.save(f'{data_dir}/time.npy', t)
        return x, x_missing, is_missing_mask, t





if __name__ == '__main__':
    gen = LinearDataGenerator()
    gen.generate_dataset()

    print('done')