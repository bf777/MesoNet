"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
from mesonet import gui_class, gui_train


def gui_start(gui_type='test'):
    if gui_type == 'test':
        gui_class.gui()
    elif gui_type == 'train':
        gui_train.gui()


if __name__ == "__main__":
    gui_start()
