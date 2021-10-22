"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the Creative Commons Attribution 4.0 International License (see LICENSE for details)
"""
from mesonet import gui_test, gui_train


def gui_start(gui_type="test", git_repo="", config_file=""):
    """
    Starts the MesoNet GUI interface.

    :param gui_type: (default: "test") Selects the type of GUI to use. Select "test" to predict atlas-to-brain
    alignments using an existing model. Select "train" to train new U-Net and DeepLabCut models.
    :param git_repo: (default: "") Manually specify the path to the MesoNet git repository instead of automatically
    locating it. You may wish to do this on if it takes a long time (e.g. > 1 min) to find the MesoNet git repository
    every time you run gui_start() (which may occur on Windows).
    :param config_file: (default: "") (Unimplemented) Specify the path to a configuration file (.yaml) to use to
    automatically fill in the GUI with desired settings.
    :return:
    """
    if gui_type == "test":
        gui_test.gui(git_repo, config_file)
    elif gui_type == "train":
        gui_train.gui()


if __name__ == "__main__":
    gui_start()
