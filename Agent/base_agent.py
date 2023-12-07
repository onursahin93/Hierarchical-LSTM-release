import torch
from torch.nn.utils import clip_grad_norm_
import random


class BaseAgent():
    def __init__(self,
                 params=None):
        self.input_size = params.input_size
        self.learning_rate = params.learning_rate
        self.batch_size = params.batch_size
        self.is_decay = params.is_decay
        self.gamma = params.gamma

        if params.use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device(params.device)
                self.n_devices = torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.n_devices = 1

    def clip_grad_norm(self):
        pass
        # clip_grad_norm_(self.online_net.parameters(), self.grad_norm_clip)

    def to_train_mode(self):
        self.network.train()

    def to_eval_mode(self):
        self.network.eval()

    def learn(self):
        pass

