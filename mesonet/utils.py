"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
import yaml
import glob
import re
import os
from os.path import join
from sys import platform


def config_project(input_dir, output_dir, mode, model_name='unet.hdf5', config='dlc/config.yaml',
                   atlas=False, sensory_match=False, sensory_path='sensory', mat_save=True, use_unet=True,
                   atlas_to_brain_align=True, olfactory_check=True, plot_landmarks=True, align_once=False,
                   original_label=False, threshold=0.0001, model='models/unet_bundary.hdf5',
                   region_labels=False, steps_per_epoch=300, epochs=60):
    """
    Generates a config file (mesonet_train_config.yaml or mesonet_test_config.yaml, depending on whether you are
    applying an existing model or training a new one).
    :param input_dir: The directory containing the input brain images
    :param output_dir: The directory containing the output files
    :param mode: If train, generates a config file for training; if test, generates a config file for applying
    the model.
    :param model_name: (optional) Set a new name for the unet model to be trained. Default is 'unet.hdf5'
    :param config: Select the config file for the DeepLabCut model to be used for landmark estimation.
    :param atlas:  Set to True to just predict the four cortical landmarks on your brain images, and not segment your
    brain images by region. Upon running mesonet.predict_dlc(config_file), MesoNet will output your brain images
    labelled with these landmarks as well as a file with the coordinates of these landmarks. Set to False to carry out
    the full brain image segmentation workflow.
    :param sensory_match: If True, MesoNet will attempt to align your brain images using peaks of sensory activation on
    sensory maps that you provide in a folder named sensory inside your input images folder. If you do not have such
    images, keep this value as False.
    :param sensory_path: If sensory_match is True, this should be set to the path to a folder containing sensory maps
    for each brain image. For each brain, put your sensory maps in a folder with the same name as the brain image (0, 1,
    2, ...).
    :param mat_save: Choose whether or not to export each predicted cortical region, each region's centrepoint, and the
    overall region of the brain to a .mat file (True = output .mat files, False = don't output .mat files).
    :param threshold:  Adjusts the sensitivity of the algorithm used to define individual brain regions from the brain
    atlas. NOTE: Changing this number may significantly change the quality of the brain region predictions; only change
    it if your brain images are not being segmented properly! In general, increasing this number causes each brain
    region contour to be smaller (less like the brain atlas); decreasing this number causes each brain region contour to
    be larger (more like the brain atlas).
    :param olfactory_check: If True, draws olfactory bulb contours on the brain image.
    :param use_unet: Choose whether or not to identify the borders of the cortex using a U-net model.
    :param atlas_to_brain_align: If True, registers the atlas to each brain image. If False, registers each brain image
    to the atlas.
    :param plot_landmarks: If True, plots DeepLabCut landmarks (large circles) and original alignment landmarks (small
    circles) on final brain image.
    :param align_once: if True, carries out all alignments based on the alignment of the first atlas and brain. This can
    save time if you have many frames of the same brain with a fixed camera position.
    :param model: The location (within the MesoNet repository) of a U-net model to be used for finding the boundaries
    of the brain region (as the default model does), or (if you have a specially trained model for this purpose)
    segmenting the entire brain into regions without the need for atlas alignment. Only choose another model if you have
    another model that you would like to use for segmenting the brain.
    :param region_labels: If True, MesoNet will attempt to label each brain region according to the Allen Institute's
    Mouse Brain Atlas. Otherwise, MesoNet will label each region with a number. Please note that this feature is
    experimental!
    :param steps_per_epoch: During U-Net training, the number of steps that the model will take per epoch. Defaults to
    300 steps per epoch.
    :param epochs: During U-Net training, the number of epochs for which the model will run. Defaults to 60 epochs (set
    lower for online learning, e.g. if augmenting existing model).
    :return config_file: The path to the config_file. If you run this function as config_file = config_project(...) then
    you can directly get the config file path to be used later.
    """

    # git_repo_base = 'C:/Users/mind reader/Desktop/mesonet/mesonet'
    git_repo_base = find_git_repo()
    print(git_repo_base)
    config = join(git_repo_base, config)
    model = join(git_repo_base, model)
    filename = "mesonet_config.yaml"
    data = dict()

    if mode == 'test':
        filename = "mesonet_test_config.yaml"
        num_images = len(glob.glob(os.path.join(input_dir, '*.png')))
        data = dict(
            config=config,
            input_file=input_dir,
            output=output_dir,
            atlas=atlas,
            sensory_match=sensory_match,
            sensory_path=sensory_path,
            mat_save=mat_save,
            threshold=threshold,
            num_images=num_images,
            model=model,
            git_repo_base=git_repo_base,
            region_labels=region_labels,
            landmark_arr=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            use_unet=use_unet,
            atlas_to_brain_align=atlas_to_brain_align,
            olfactory_check=olfactory_check,
            plot_landmarks=plot_landmarks,
            align_once=align_once,
            original_label=original_label
        )
    elif mode == 'train':
        filename = "mesonet_train_config.yaml"
        data = dict(
            input_file=input_dir,
            model_name=model_name,
            log_folder=output_dir,
            git_repo_base=git_repo_base,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            bodyparts=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        )

    with open(os.path.join(output_dir, filename), 'w') as outfile:
        yaml.dump(data, outfile)

    config_file = os.path.join(output_dir, filename)
    return config_file


def parse_yaml(config_file):
    """
    Parses the config file and returns a dictionary with its parameters.
    :param config_file: The full path to a MesoNet config file (generated using mesonet.config_project())
    """
    with open(config_file, 'r') as stream:
        try:
            d = yaml.safe_load(stream)
            return d
        except yaml.YAMLError as exc:
            print(exc)


def natural_sort_key(s):
    """
    Alphanumeric sort workaround from
    https://stackoverflow.com/questions/19366517/sorting-in-python-how-to-sort-a-list-containing-alphanumeric-values

    :param s: The list to be sorted.
    :return: Naturally sorted version of list.
    """
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def find_git_repo():
    # Preferred (faster) option to find mesonet git repo is to set it as an environment variable
    git_repo_base = ''
    try:
        git_repo_base = os.environ['MESONET_GIT']
    except:
        # If we can't find the environment variable, search for the mesonet git repository
        # solution to find git repository on computer adapted from:
        # https://stackoverflow.com/questions/5153317/python-how-to-do-a-system-wide-search-for-a-file-when-just-the-filename-not-pa

        root_folder = 'C:\\'
        git_repo_marker = "mesonet.txt"
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            # linux or mac
            root_folder = '/home'
        elif platform == "win32":
            # Windows
            root_folder = 'C:\\'
        for root, dirs, files in os.walk(root_folder):
            if git_repo_marker in files:
                git_repo_base = root
                break
    return git_repo_base
