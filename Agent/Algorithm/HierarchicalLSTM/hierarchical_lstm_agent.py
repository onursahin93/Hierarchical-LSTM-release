import torch
from Agent.base_agent import BaseAgent
from Agent.Models.HierarchicalLSTM.hierarchical_lstm_model import HierarchicalLSTMModel

class HierarchicalLSTMAgent(BaseAgent):
    def __init__(self, params):
        BaseAgent.__init__(self, params)

        self.network = HierarchicalLSTMModel(params).to(device=self.device)
        # self.learning_rate = params.learning_rate
        self.window_length = params.window_length
        self.window_input = []#[None for _ in range(self.window_length+1)]
        self.window_pattern = []
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.different_learning_rates_for_lstms = params.different_learning_rates_for_lstms

        if not self.different_learning_rates_for_lstms:
            self.optimizer = getattr(torch.optim, f"{params.optimizer}")(self.network.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = getattr(torch.optim, f"{params.optimizer}")(
                [{"params": p, "lr": ("lstm_list" in n)*self.window_length*self.learning_rate + (1 -("lstm_list" in n))*self.learning_rate} for n, p in self.network.named_parameters()], lr=self.learning_rate)
        if self.is_decay:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)


    def __call__(self, x, mask):
        y_hat = None
        self.window_input.append(x)
        self.window_pattern.append(mask)
        if len(self.window_input) < self.window_length+1:
            pass
        else:
            if len(self.window_input) == self.window_length+2:
                self.window_input.pop(0)
                self.window_pattern.pop(0)
            # self.network(self.window_input, self.window_pattern)
            y_hat = self.network(torch.stack(tuple(self.window_input)).to(dtype=torch.float32, device=self.device),
                                 torch.stack(tuple(self.window_pattern)).to(dtype=torch.float32, device=self.device))
        return y_hat


    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.detach_hiddens()
        if self.is_decay:
            self.scheduler.step()


    def detach_hiddens(self):
        self.network.detach_hiddens()

    def reset(self):
        self.network.reset_lstm_hiddens()




if __name__ == "__main__":
    raise NotImplementedError