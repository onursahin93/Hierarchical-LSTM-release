import torch
from Interaction.base_interaction import BaseInteraction
from tqdm import trange
from Logger.vanilla_logger import VanillaLogger
from utils.utils import print_and_log
import time
class Interaction(BaseInteraction):
    def __init__(self, params):
        BaseInteraction.__init__(self, params)

        self.num_epoch = params.num_epoch
        self.evaluation_interval = params.evaluation_interval
        self.loss_fn = torch.nn.MSELoss()

        self.logger = VanillaLogger(params=params)
        # self.logger.log_hparams(params=params)



    def interact(self, model, data, params):
        if params.is_train:
            self.interact_train(model, data, params)


    def interact_train(self, model, data, params):
        print("training starts (batch)")
        sum_y_mask_train = data["train"]["y_mask"].sum().to(dtype=torch.float32)
        self.interact_test(model, data, params, 0)
        model.to_train_mode()
        for epoch_idx in trange(1, params.num_epoch + 1):
            initial_time = time.time()
            epoch_loss_train = 0
            loss= 0
            for t in range(data["train"]["x"].shape[0]):
                y_hat = model(data["train"]["x"][t], data["train"]["x_mask"][t])

                if (not y_hat is None) and data["train"]["y_mask"][t]:
                    loss += self.loss_fn(y_hat, data["train"]["y"][t].unsqueeze(0))
                    epoch_loss_train += loss.detach().cpu().numpy()
            model.learn(loss)
            model.reset()
            print_and_log(params.log_file, f"\nEpoch :{epoch_idx}, Train Loss: {epoch_loss_train}, Epoch duration: {time.time() - initial_time}")
            self.logger.log(label="train_loss", value=epoch_loss_train, step=epoch_idx, main_tag="train-epoch")
            self.logger.log(label="train_loss_mean", value=epoch_loss_train/sum_y_mask_train, step=epoch_idx, main_tag="train-epoch")

            if epoch_idx % self.evaluation_interval == 0:
                self.interact_test(model, data, params, epoch_idx)


    def interact_test(self, model, data, params, epoch_idx):
        sum_y_mask_test = data["test"]["y_mask"].sum().to(dtype=torch.float32)
        model.to_eval_mode()
        model.reset()
        initial_time = time.time()
        epoch_loss_test = 0
        with torch.no_grad():
            for t in range(data["test"]["x"].shape[0]):
                y_hat = model(data["test"]["x"][t], data["test"]["x_mask"][t])

                if (not y_hat is None) and data["test"]["y_mask"][t]:
                    loss = self.loss_fn(y_hat, data["test"]["y"][t].unsqueeze(0))
                    epoch_loss_test += loss.detach().cpu().numpy()
            model.reset()
            # print(f"Epoch :{epoch_idx}, Train Loss: {epoch_loss_test}, Epoch duration: {time.time() - initial_time}")
            print_and_log(params.log_file, f"Epoch :{epoch_idx}, Test Loss: {epoch_loss_test}, Epoch duration: {time.time() - initial_time}")
            self.logger.log(label="test_loss", value=epoch_loss_test, step=epoch_idx, main_tag="test-epoch")
            self.logger.log(label="test_loss_mean", value=epoch_loss_test/sum_y_mask_test, step=epoch_idx, main_tag="test-epoch")
            # self.logger.log_hparams(params, 3)

        model.reset()
        model.to_train_mode()




if __name__ ==  "__main__":
    import torch
    from Config.config_manager import ConfigManager
    from Data.DataReader.NYSE.nyse_reader import NYSEDataReader
    from Agent.agent_factory import AgentFactory
    import time
    params = ConfigManager().args

    reader = NYSEDataReader(params=params)
    data = reader.get_missing_data()

    model = AgentFactory().get_agent(params)
    loss_fn = torch.nn.MSELoss()

    for ep in range(100):
        initial_time = time.time()
        total_loss = 0
        for t in range(data["train"]["x"].shape[0]):
            y_hat = model(data["train"]["x"][t], data["train"]["x_mask"][t])

            if (not y_hat is None) and data["train"]["y_mask"][t]:
                loss = loss_fn(y_hat, data["train"]["y"][t].unsqueeze(0))
                total_loss += loss.detach().cpu().numpy()
                model.learn(loss)
        model.reset()
        print(f"Epoch :{ep}, Test Loss: {total_loss}, Epoch duration: {time.time() - initial_time}")

        # print(time.time() - initial_time)






    print("")