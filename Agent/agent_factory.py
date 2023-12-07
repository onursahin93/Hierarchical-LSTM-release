from Agent.Algorithm.HierarchicalLSTM.hierarchical_lstm_agent import HierarchicalLSTMAgent

class AgentFactory():

    def get_agent(self, params):
        agent_type = params.algorithm
        if agent_type == 'HierarchicalLSTM':
            from Agent.Algorithm.HierarchicalLSTM.hierarchical_lstm_agent import HierarchicalLSTMAgent
            return HierarchicalLSTMAgent(params)

        else:
            raise  NotImplementedError