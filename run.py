from Config.config_manager import ConfigManager
from Data.DataReader.data_reader_factory import DataReaderFactory
from Agent.agent_factory import AgentFactory
# from Interaction.interaction import Interaction
from Interaction.interaction_batch import Interaction
from utils.set_seed import set_seed

params = ConfigManager().args
set_seed(params)

reader = DataReaderFactory().get_data_reader(params=params)
data = reader.get_missing_data()

model = AgentFactory().get_agent(params)
interaction = Interaction(params=params)

interaction.interact(model=model,
                     data=data,
                     params=params)
