import torch

class BaseDataReader():
    def __init__(self, params):
        self.is_batch = params.is_batch
        self.is_normalize = params.is_normalize


    def normalize(self):
        b = torch.max(self.raw, 0)[0] + 0.01
        a = torch.min(self.raw, 0)[0] - 0.01

        self.raw = (self.raw - (a + b) / 2) * 2 / (b - a)
