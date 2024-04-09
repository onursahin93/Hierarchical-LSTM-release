import numpy as np
# import pandas as pd
import warnings
import torch
from argparse import Namespace
from Config.constants import MAIN_DIR
from Data.DataReader.base_data_reader import BaseDataReader
import random



class KinematicsDataReader(BaseDataReader):
    def __init__(self,
                 params):
        super(KinematicsDataReader, self).__init__(params=params)
        self.device = params.device
        self.data_path = f"{MAIN_DIR}/Data/old_data/Kinematics/kin8nm.data"
        self.label_index = 8
        self.n_step = 0#params.n_step
        self.require_presence_pattern = False
        self.raw = torch.from_numpy(np.genfromtxt(self.data_path, delimiter=','))
        if self.is_normalize:
            self.normalize()
        self.train_ratio = params.train_ratio
        self.missing_ratio = params.missing_ratio

        self.train_length = round((self.raw.shape[0]*self.train_ratio))
        self.test_length = self.raw.shape[0] - self.train_length - self.n_step

        self.x_t_train = self.raw[:self.train_length,:8]
        self.x_t_test = self.raw[self.train_length:-self.n_step, :8] if self.n_step > 0 else self.raw[self.train_length:, :8]
        self.y_t_train = (self.raw[self.n_step:self.train_length + self.n_step, 8]).unsqueeze(1)
        self.y_t_test = (self.raw[self.train_length + self.n_step:, 8]).unsqueeze(1)

        self.raw_mask = torch.rand(self.raw.shape[0]) > self.missing_ratio
        self.x_mask_train = self.raw_mask[:self.train_length]
        self.y_mask_train = self.raw_mask[self.n_step:self.train_length + self.n_step]
        self.x_mask_test = self.raw_mask[self.train_length:-self.n_step] if self.n_step > 0 else self.raw_mask[self.train_length:]
        self.y_mask_test = self.raw_mask[self.train_length + self.n_step:]

        self.always_true_mask = torch.rand(self.raw.shape[0]) > -1
        if self.missingness_type == "MCAR":
            self.raw_mask = torch.rand(self.raw.shape[0]) > self.missing_ratio

        elif self.missingness_type == "MNAR_v0":
            """
                Missingness related to the unobserved data
            """
            # MNAR begin
            raw_label = self.raw[:, 8]
            sorted_raw_label = np.sort(raw_label)
            low_value = sorted_raw_label[int(np.round(self.raw.shape[0] * self.missing_ratio))]
            high_value = sorted_raw_label[int(np.round(self.raw.shape[0] * (1 - self.missing_ratio)))]

            mask_list = []
            for idx in range(self.raw.shape[0]):
                if raw_label[idx] < low_value or raw_label[idx] > high_value:
                    if random.random() > 0.5:
                        mask_list.append(0)
                    else:
                        mask_list.append(1)
                else:
                    mask_list.append(1)

            self.raw_mask = torch.tensor(mask_list, dtype=torch.bool)

        elif self.missingness_type == "MNAR":
            # completely delete < low or > high
            # MNAR begin
            raw_label = self.raw[:, 8]
            sorted_raw_label = np.sort(raw_label)
            low_value = sorted_raw_label[int(np.round(self.raw.shape[0] * self.missing_ratio * 0.5))]
            high_value = sorted_raw_label[int(np.round(self.raw.shape[0] * (1 - self.missing_ratio * 0.5)))]

            mask_list = []
            for idx in range(self.raw.shape[0]):
                if raw_label[idx] < low_value or raw_label[idx] > high_value:
                    if random.random() > -1:  # always True
                        mask_list.append(0)
                    else:
                        mask_list.append(1)
                else:
                    mask_list.append(1)

            # self.raw_mask = torch.rand(self.raw.shape[0]) > self.missing_ratio # MCAR
            self.raw_mask = torch.tensor(mask_list, dtype=torch.bool)

        elif self.missingness_type == "MAR":
            """
                Missingness related to the observed data
            """
            # MAR begins

            missingness_done = False
            raw_label = self.raw[:, 8]
            mask_list = []
            mask_np = np.ones(self.raw.shape[0])

            max_delete_window = 3
            sorted_raw_label = np.sort(raw_label)
            low_value = sorted_raw_label[int(np.round(self.raw.shape[0] * self.missing_ratio / 2))]
            high_value = sorted_raw_label[int(np.round(self.raw.shape[0] * (1 - self.missing_ratio / 2)))]
            while not missingness_done:
                for idx in range(self.raw.shape[0]):
                    if mask_np[idx]:  # if data idx exists
                        if raw_label[idx] < low_value or raw_label[idx] > high_value:
                            for next_idx in range(idx + 1, idx + max_delete_window + 1):  # For example: idx=5, max_delete_window=3, next_idx = 6 7 8
                                if (next_idx < (self.raw.shape[0])) and (mask_np[next_idx] == True):  # next_idx is in valid range and exists
                                    if random.random() > 0.75:
                                        mask_np[next_idx] = 0
                                        break

                        if (1 - (mask_np.sum() / self.raw.shape[0])) >= self.missing_ratio:
                            missingness_done = True
                            print("Done")
                            break

            print(mask_np.sum() / self.raw.shape[0])
            self.raw_mask = torch.from_numpy(mask_np).to(dtype=torch.bool)
            # MAR ends

        self.x_mask_train = self.raw_mask[:self.train_length]
        if params.is_y_always_exists:
            self.y_mask_train = self.always_true_mask[self.n_step:self.train_length + self.n_step]
        else:
            self.y_mask_train = self.raw_mask[self.n_step:self.train_length + self.n_step]

        self.x_mask_test = self.raw_mask[self.train_length:-self.n_step] if self.n_step > 0 else self.raw_mask[self.train_length:]
        if params.is_y_always_exists:
            self.y_mask_test = self.always_true_mask[self.train_length + self.n_step:]
        else:
            self.y_mask_test = self.raw_mask[self.train_length + self.n_step:]


        self.calculate_deltas()
        if params.algorithm == "HierarchicalLSTM":
            self.require_presence_pattern = True
            self.window_length = params.window_length
            self.calculate_presence_patterns()
        print("")


    def calculate_deltas(self):
        self.delta_train = torch.zeros(size=[self.train_length])
        self.delta_test = torch.zeros(size=[self.test_length])

        for t in range(self.train_length):
            if t == 0:
                self.delta_train[t] = 0
            else:
                if self.x_mask_train[t-1]:
                    self.delta_train[t] = 1
                else:
                    self.delta_train[t] = 1 + self.delta_train[t-1]

        for t in range(self.test_length):
            if t == 0:
                self.delta_test[t] = 0
            else:
                if self.x_mask_test[t - 1]:
                    self.delta_test[t] = 1
                else:
                    self.delta_test[t] = 1 + self.delta_test[t - 1]

    def calculate_presence_patterns(self):
        self.presence_pattern_train = torch.zeros(size=[self.train_length, self.window_length])
        self.presence_pattern_test = torch.zeros(size=[self.test_length, self.window_length])
        for t in range(self.train_length):
            if t < self.window_length:
                pass
            else:
                self.presence_pattern_train[t] = self.x_mask_train[t-self.window_length+1:t+1]

        for t in range(self.test_length):
            if t < self.window_length:
                pass
            else:
                self.presence_pattern_test[t] = self.x_mask_test[t-self.window_length+1:t+1]


    def read_raw_data(self):
        ratio = np.loadtxt(self.data_path)
        raw = ratio.cumprod(axis=0)
        return raw

    def read_missing_data(self, missing_ratio=None):
        raise NotImplementedError
        if missing_ratio is None:
            warnings.warn("Missing Ratio is not entered. Accepted as 0.")
            missing_ratio = 0.0

        L = self.raw.shape[0]
        mask = torch.rand(L) > missing_ratio
        # print(mask.float().mean())
        indices = torch.arange(L)
        raw= torch.from_numpy(self.raw)


        print("")
        # ratio = np.loadtxt(self.data_path)
        # raw = ratio.cumprod(axis=0)
        return raw

    def get_missing_data(self):
        data = {"train": {"x": self.x_t_train.to(dtype=torch.float32, device=self.device),
                          "y": self.y_t_train.to(dtype=torch.float32, device=self.device),
                          "x_mask": self.x_mask_train,
                          "y_mask": self.y_mask_train},
                "test":  {"x": self.x_t_test.to(dtype=torch.float32, device=self.device),
                          "y": self.y_t_test.to(dtype=torch.float32, device=self.device),
                          "x_mask": self.x_mask_test,
                          "y_mask": self.y_mask_test}}

        if self.require_presence_pattern:
            data["train"]["presence_pattern"] = self.presence_pattern_train
            data["test"]["presence_pattern"] = self.presence_pattern_test

        return data





if __name__ == "__main__":
    params = Namespace(**{"train_ratio": 0.6,
                          "device": "cuda:0",
                          "missing_ratio": 0.2,
                          "n_step": 1,
                          "algorithm": "HierarchicalLSTM",
                          "window_length": 3})
    reader = KinematicsDataReader(params=params)
    # data = reader.read_missing_data(missing_ratio=0.6)
    data = reader.get_missing_data()


    print("")