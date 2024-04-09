import torch

class BaseDataReader():
    def __init__(self, params):
        self.is_batch = params.is_batch
        self.is_normalize = params.is_normalize

        self.missingness_type = params.missingness_type

        # TEMP
        mnar_implemented_list = ["Kinematics"]
        mar_implemented_list = ["Kinematics"]

        if self.missingness_type in ["MNAR_v0", "MNAR"]:
            if not params.dataset in mnar_implemented_list:
                raise NotImplementedError(f"{params.dataset} has no MNAR missingness implementation.")

        elif self.missingness_type in ["MAR"]:
            if not params.dataset in mar_implemented_list:
                raise NotImplementedError(f"{params.dataset} has no MAR missingness implementation.")

    def normalize(self):
        b = torch.max(self.raw, 0)[0] + 0.01
        a = torch.min(self.raw, 0)[0] - 0.01

        self.raw = (self.raw - (a + b) / 2) * 2 / (b - a)
