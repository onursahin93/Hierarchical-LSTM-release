import torch
import torch.nn as nn
from torch.autograd import Variable


class BaseModel(nn.Module):
    def __init__(self,
                 params=None):
        super(BaseModel, self).__init__()

        self.input_size = params.input_size
        self.rnn_hidden_size = params.rnn_hidden_size
        self.out_size = params.out_size if hasattr(params, "out_size") else 999
        self.sequence_length = params.sequence_length if hasattr(params, "sequence_length") else None
        self.batch_size = params.batch_size if hasattr(params, "batch_size") else 1
        self.device = params.device if hasattr(params, "device") else "cuda:0"
        self.init_type = params.init
        if self.init_type == "normal":
            self.normal_std = params.normal_std

    def init_weights(self):
        if self.init_type == "default":
            pass
        elif self.init_type == "normal":
            for name, param in self.named_parameters():
                torch.nn.init.normal_(param, mean=0., std=self.normal_std)
        elif self.init_type == "uniform":
            for name, param in self.named_parameters():
                torch.nn.init.uniform_(param, a=-1.0, b=1.0)
        elif self.init_type == "xavier_normal":
            for name, param in self.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_normal_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)
        elif self.init_type == "xavier_uniform":
            for name, param in self.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)
        else:
            raise NotImplementedError

    def detach_hiddens(self):
        self.c.detach_()
        self.h.detach_()

    def reset_lstm_hiddens(self):
        self.c = Variable(torch.zeros(self.batch_size, self.rnn_hidden_size, device=self.device))
        self.h = Variable(torch.zeros(self.batch_size, self.rnn_hidden_size, device=self.device))
