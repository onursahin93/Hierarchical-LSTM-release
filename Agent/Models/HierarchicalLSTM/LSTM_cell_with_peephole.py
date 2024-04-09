import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from math import sqrt

class LSTMCellDefault(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCellDefault, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

class LSTMCellPeephole(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCellPeephole, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

        # Peephole connections
        m = torch.eye(hidden_size, hidden_size)
        self.register_buffer('m', m)

        self.W_peephole_i = Parameter(torch.randn(hidden_size, hidden_size))
        self.W_peephole_f = Parameter(torch.randn(hidden_size, hidden_size))
        self.W_peephole_o = Parameter(torch.randn(hidden_size, hidden_size))

        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, a=-1/(sqrt(hidden_size)), b=1/(sqrt(hidden_size)))

    def forward(self, input, state):
        """
        ingate: input gate, i
        forgetgate: forget gate, f
        cellgate: block input, z
        outgate: output gate, o
        """

        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state

        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = ingate + F.linear(cx, self.W_peephole_i * self.m) # uses c_t-1
        forgetgate = forgetgate + F.linear(cx, self.W_peephole_f * self.m) # uses c_t-1


        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        # outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)

        outgate = outgate + F.linear(cy, self.W_peephole_f * self.m) # uses c_t
        outgate = torch.sigmoid(outgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

if __name__ == "__main__":
    # Test
    input_size = 2
    hidden_size = 3
    lstm_cell_peephole = LSTMCellPeephole(input_size=input_size, hidden_size=hidden_size)
    for name, param in lstm_cell_peephole.named_parameters():
        print(name)
        torch.nn.init.normal_(param, mean=0., std=0.1)

    x = torch.randn(1, input_size)

    hx = torch.randn(1, hidden_size)  # (batch, hidden_size)
    cx = torch.randn(1, hidden_size)

    hy, cy = lstm_cell_peephole(input=x, state=(hx, cx))
    print()