import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import copy
from Agent.Models.base_model import BaseModel
from Agent.Models.HierarchicalLSTM.LSTM_cell_with_peephole import LSTMCellPeephole


class HierarchicalLSTMModel(BaseModel):
    def __init__(self, params):
                 # input_size, hidden_size, window_length, out_size=1, batch_size = 1, device='cuda:0'):
        """
        HierarchicalLSTM:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            window_length: dimension of window length for presence vector
        """

        super(HierarchicalLSTMModel, self).__init__(params)

        # self.device = device
        self.window_length = params.window_length
        self.lstm_number = 2**self.window_length
        self.use_presence_patterns = params.use_presence_patterns
        self.is_fc_tilde_different = params.is_fc_tilde_different
        self.temperature = torch.tensor([params.temperature], device=self.device, dtype=torch.float32)
        self.is_peephole = params.is_peephole
        # self.hidden_size = hidden_size
        # self.input_size = input_size
        # self.batch_size = batch_size
        if not self.is_peephole:  # default case, without peephole connections
            self.lstm_list = nn.ModuleList([torch.nn.LSTMCell(input_size=self.input_size, hidden_size=self.rnn_hidden_size, bias=True) for _ in range(self.lstm_number)])
        else:
            self.lstm_list = nn.ModuleList([LSTMCellPeephole(input_size=self.input_size, hidden_size=self.rnn_hidden_size) for _ in range(self.lstm_number)])
        self.lstm_c_list, self.lstm_h_list = [], []
        # self.init_lstm_list(lstm_number=self.lstm_number)
        self.reset_lstm_hiddens()
        self.init_lstm_id_2_presence_list()

        self.fc_tilde_size = self.rnn_hidden_size + self.use_presence_patterns*2*self.window_length
        if self.is_fc_tilde_different:
            self.FC_tilde_list = nn.ModuleList([nn.Linear(in_features=self.fc_tilde_size, out_features=1) for _ in range(self.lstm_number)])
        else:
            self.FC_tilde = nn.Linear(in_features=self.fc_tilde_size, out_features=1)
        self.FC_hat = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.out_size)
        self.init_weights()


    # def init_lstm_weights(self):
    #     for name, param in self.lstm_list.named_parameters():
    #         torch.nn.init.normal_(param)
    #
    # def init_FC_weights(self):
    #     for name, param in self.named_parameters():
    #         torch.nn.init.normal_(param)


    def get_valid_lstm_ids(self, presence):

        if not isinstance(presence, torch.Tensor):
            presence = torch.tensor(presence)

        if presence.shape[0] != self.window_length:
            raise RuntimeError("Presence length should be equal to window length")
        valid_lstm_idx = []
        for lstm_id in range(self.lstm_number):
            if (presence >= self.lstm_id_2_presence_list[lstm_id]).all():
                valid_lstm_idx.append(lstm_id)
        return valid_lstm_idx

    def get_valid_mask(self, lstm_id, presence):
        mask = torch.logical_and(presence, self.lstm_id_2_presence_list[lstm_id])
        return mask

    # def get_valid_input_sequence(self, window_input, valid_mask):
    #     return torch.masked_select(window_input, valid_mask)

    def get_valid_input_sequence(self, window_input, lstm_id, presence):
        valid_mask = self.get_valid_mask(lstm_id, presence)
        return window_input.index_select(index=torch.masked_select(torch.arange(window_input.shape[0], device=self.device), valid_mask.squeeze(0)), dim=0)
        # return torch.masked_select(window_input, valid_mask)

    def init_lstm_id_2_presence_list(self):
        # pass
        self.lstm_id_2_presence_list = []
        for lstm_id in range(self.lstm_number):
            self.lstm_id_2_presence_list.append(self.dec2bin(torch.tensor([lstm_id], device=self.device)))


    def reset_lstm_hiddens(self):
        self.lstm_c_list = []
        self.lstm_h_list = []
        for i in range(self.lstm_number):
            self.lstm_c_list.append(Variable(torch.zeros(self.batch_size, self.rnn_hidden_size, device=self.device)))
            self.lstm_h_list.append(Variable(torch.zeros(self.batch_size, self.rnn_hidden_size, device=self.device)))

    def init_lstm_list(self, lstm_number):
        for i in range(lstm_number):
            self.lstm_list.append(torch.nn.LSTMCell(input_size=self.input_size, hidden_size=self.rnn_hidden_size, bias=True))
        # self.lstm_list = nn.ModuleList(self.lstm_list)

    def detach_hiddens(self):
        for i in range(self.lstm_number):
            self.lstm_c_list[i].detach_()
            self.lstm_h_list[i].detach_()



    # @staticmethod
    def dec2bin(self, x):
        # bits = self.window_length
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(self.window_length - 1, -1, -1).to(x.device, x.dtype)
        # mask = 2 ** torch.arange(self.window_length - 1, -1, -1).to(self.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    # @staticmethod
    def bin2dec(self, b):
        # bits = self.window_length
        mask = 2 ** torch.arange(self.window_length - 1, -1, -1).to(self.device, b.dtype)
        return torch.sum(mask * b, -1)

    def __check_forward_step(self, x_list):
        if (x_list ==  torch.tensor([999999], device=self.device)).any():
            raise RuntimeError('A missing input is processing.')

    def pass_hiddens_from_main_to_leaves(self, valid_leaf_lstm_idx):
        for leaf_lstm_id in valid_leaf_lstm_idx:
            self.lstm_c_list[leaf_lstm_id] = self.lstm_c_list[0].clone()
            self.lstm_h_list[leaf_lstm_id] = self.lstm_h_list[0].clone()

    def weighted_sum_hiddens(self, valid_lstm_idx, window_input_mask):
        if self.is_fc_tilde_different:
            valid_h = torch.index_select(torch.cat(self.lstm_h_list, dim=0), dim=0, index=torch.tensor(valid_lstm_idx, device=self.device))
            # combination_energies = torch.empty(size=[len(valid_lstm_idx), 1], device=self.device, dtype=torch.float32)
            combination_energies = []
            for valid_idx in valid_lstm_idx:
                if self.use_presence_patterns:
                    valid_h_bar_idx = torch.cat([self.lstm_h_list[valid_idx],
                                                 self.lstm_id_2_presence_list[valid_idx],
                                                 window_input_mask.unsqueeze(0)],
                                                dim=1)

                    combination_energies.append(self.FC_tilde_list[valid_idx](valid_h_bar_idx))

                else:
                    raise NotImplementedError

            # combination_weights = nn.functional.softmax(torch.cat(combination_energies), dim=0)
            combination_weights = nn.functional.softmax(self.temperature*torch.cat(combination_energies), dim=0)
            return torch.matmul(combination_weights.t(), valid_h)

        else:
            valid_h = torch.index_select(torch.cat(self.lstm_h_list, dim=0), dim=0, index=torch.tensor(valid_lstm_idx, device=self.device))
            if self.use_presence_patterns:
                valid_lstm_presence_pattern = torch.index_select(torch.cat(self.lstm_id_2_presence_list), index=torch.tensor(valid_lstm_idx, device=self.device), dim=0)
                valid_h_bar = torch.cat([valid_h,
                           valid_lstm_presence_pattern,
                           window_input_mask.expand([valid_lstm_presence_pattern.shape[0],valid_lstm_presence_pattern.shape[1]])],
                          dim=1)

                combination_weights = nn.functional.softmax(self.FC_tilde(valid_h_bar), dim=0)
            else:
                combination_weights = nn.functional.softmax(self.FC_tilde(valid_h), dim=0)
            # combined h
            return torch.matmul(combination_weights.t(), valid_h)

    def step(self, x_list, lstm_id):
        self.__check_forward_step(x_list)
        # hx, cx = rnn(input[i], (hx, cx))
        hx, cx = self.lstm_h_list[lstm_id], self.lstm_c_list[lstm_id]
        for i in range(x_list.shape[0]):
            hx, cx = self.lstm_list[lstm_id](x_list[i].unsqueeze(0), (hx, cx))
            # hx, cx = self.lstm_list[lstm_id](x_list[i].unsqueeze(0).unsqueeze(1), (hx, cx))

        self.lstm_h_list[lstm_id], self.lstm_c_list[lstm_id] = hx, cx

    def clone(self):
        return copy.deepcopy(self)

    def forward(self, x_seq, presence_mask):
        """
        :param x_seq: [main_lstm_input, window_inputs] -> window_length + 1
        :param presence_mask: [main_lstm_input_mask, window_input_masks] -> window_length + 1
        :return:
        """

        main_input = x_seq[0]
        main_input_mask = presence_mask[0]
        window_input = x_seq[1:]
        window_input_mask = presence_mask[1:]

        valid_lstm_idx = self.get_valid_lstm_ids(window_input_mask)
        valid_leaf_lstm_idx = [id for id in valid_lstm_idx if id not in [0]]

        if main_input_mask:
            # print('main_input exists')
            self.step(main_input.unsqueeze(0), lstm_id=0)



        self.pass_hiddens_from_main_to_leaves(valid_leaf_lstm_idx)

        for lstm_id in valid_leaf_lstm_idx:
            valid_window_inputs = self.get_valid_input_sequence(window_input, lstm_id, window_input_mask)
            self.step(valid_window_inputs, lstm_id)


        combined_hidden = self.weighted_sum_hiddens(valid_lstm_idx, window_input_mask)
        output = self.FC_hat(combined_hidden)
        return output



if __name__ == '__main__':
    print("done")