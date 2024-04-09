# Hierarchical-LSTM
This repository is the implementation of Hierarchical-LSTM architecture.

## Requirements
torch==1.13.1 is used in the implementation.

## Code Structure
- Config folder manages the argument operations. config_manager.py is the main part of the Config folder, which combines the other config files.
- Agent folder contains the implementation of the algorithm. Agent/Algorithm/HierarchicalLSTM/hierarchical_lstm_agent.py includes the functions of Hierarchical-LSTM such as learn(), \_\_call__, detach_hiddens() as well as the HierarchicalLSTMModel as its network. Agent/Models/HierarchicalLSTM/hierarchical_lstm_model.py contains the network of the Hierarchical-LSTM architecture.
- Data folder contains the files for reading and preparing datasets.
- Interaction folder contains the training loop, which also conducts the logging operations.

## Usage 

- python run.py --run-id 1234 --dataset $DATASET --device cuda:0 --optimizer $OPTIMIZER --learning-rate $LR --algorithm HierarchicalLSTM --rnn-hidden-size 8 --missing-ratio $MISSING_RATIO --gamma 1 --window-length 3 --temperature 1 --different-learning-rates-for-lstms False

- python run.py --run-id 1234 --dataset Kinematics --device cuda:0 --optimizer Adam --learning-rate 0.001 --algorithm HierarchicalLSTM --rnn-hidden-size 8 --missing-ratio 0.5 --gamma 1 --window-length 3 --temperature 1 --different-learning-rates-for-lstms True

