import os
curent_dir = os.path.dirname((__file__))
parent = os.path.dirname(curent_dir)
MAIN_DIR = parent
# MAIN_DIR = "C:/Users/onurs/PycharmProjects/HierarchicalLSTM"
print(f'Your main directory is : {MAIN_DIR}')
eps = 1e-5