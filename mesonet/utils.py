"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the Creative Commons Attribution 4.0 International License (see LICENSE for details)
"""
import yaml
import glob
import re
import os
from os.path import join
from sys import platform
import scipy.io as sio
import cv2
import numpy as np
import neurite as ne
import matplotlib.pyplot as plt
import pathlib


def config_project(
    input_dir,
    output_dir,
    mode,
    model_name="unet.hdf5",
    config="dlc/config.yaml",
    atlas=False,
    sensory_match=False,
    sensory_path="sensory",
    mask_generate=True,
    mat_save=True,
    use_unet=True,
    use_dlc=True,
    atlas_to_brain_align=True,
    olfactory_check=True,
    plot_landmarks=True,
    align_once=False,
    original_label=False,
    use_voxelmorph=True,
    exist_transform=False,
    voxelmorph_model="motif_model_atlas.h5",
    template_path="templates",
    flow_path="",
    coords_input_file="",
    atlas_label_list=[],
    threshold=0.0001,
    model="models/unet_bundary.hdf5",
    region_labels=False,
    steps_per_epoch=300,
    epochs=60,
    rotation_range=0.3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
):
    """
    Generates a config file (mesonet_train_config.yaml or mesonet_test_config.yaml, depending on whether you are
    applying an existing model or training a new one).

    :param input_dir: (required) The directory containing the input brain images
    :param output_dir: (required) The directory containing the output files
    :param mode: (required) If train, generates a config file for training; if test, generates a config file for applying
    the model.
    :param model_name: (default = 'unet.hdf5') Set a new name for the unet model to be trained. Default is 'unet.hdf5'
    :param config: (default = 'dlc/config.yaml') Select the config file for the DeepLabCut model to be used for landmark estimation.
    :param atlas: (default = False) Set to True to just predict the four cortical landmarks on your brain images, and not segment your
    brain images by region. Upon running mesonet.predict_dlc(config_file), MesoNet will output your brain images
    labelled with these landmarks as well as a file with the coordinates of these landmarks. Set to False to carry out
    the full brain image segmentation workflow.
    :param sensory_match: (default = False) If True, MesoNet will attempt to align your brain images using peaks of sensory activation on
    sensory maps that you provide in a folder named sensory inside your input images folder. If you do not have such
    images, keep this value as False.
    :param sensory_path: (default = 'sensory') If sensory_match is True, this should be set to the path to a folder containing sensory maps
    for each brain image. For each brain, put your sensory maps in a folder with the same name as the brain image (0, 1,
    2, ...).
    :param mask_generate: (default = True) If mask_generate is True, running the function predict_regions will only
    generate U-Net based masks without an atlas-to-brain alignment.
    :param mat_save: (default = True) Choose whether or not to export each predicted cortical region, each region's centrepoint, and the
    overall region of the brain to a .mat file (True = output .mat files, False = don't output .mat files).
    :param threshold: (default = True) Adjusts the sensitivity of the algorithm used to define individual brain regions from the brain
    atlas. NOTE: Changing this number may significantly change the quality of the brain region predictions; only change
    it if your brain images are not being segmented properly! In general, increasing this number causes each brain
    region contour to be smaller (less like the brain atlas); decreasing this number causes each brain region contour to
    be larger (more like the brain atlas).
    :param olfactory_check: (default = True) If True, draws olfactory bulb contours on the brain image.
    :param use_unet: (default = True) Choose whether or not to identify the borders of the cortex using a U-net model.
    :param use_dlc: (default = True) Choose whether or not to try and register the atlas and brain image using a DeepLabCut model.
    :param atlas_to_brain_align: (default = True) If True, registers the atlas to each brain image. If False, registers each brain image
    to the atlas.
    :param plot_landmarks: (default = True) If True, plots DeepLabCut landmarks (large circles) and original alignment landmarks (small
    circles) on final brain image.
    :param align_once: (default = False) If True, carries out all alignments based on the alignment of the first atlas and brain. This can
    save time if you have many frames of the same brain with a fixed camera position.
    :param original_label: (default = False) If True, uses a brain region labelling approach that attempts to automatically sort brain
    regions in a consistent order (left to right by hemisphere, then top to bottom for vertically aligned regions). This
    approach may be more flexible if you're using a custom brain atlas (i.e. not one in which region is filled with a
    unique number).
    :param use_voxelmorph: (default = True) Choose whether or not to apply a local deformation registration for image registration,
    using a voxelmorph model.
    :param exist_transform: (default = False) If True, uses an existing voxelmorph transformation field for all data instead of predicting
    a new transformation.
    :param voxelmorph_model: (default = 'motif_model_atlas.h5') The name of a .h5 model located in the models folder of the git repository for MesoNet,
    generated using voxelmorph and containing weights for a voxelmorph local deformation model.
    :param template_path: (default = 'templates') The path to a template atlas (.npy or .mat) to which the brain image will be aligned in
    voxelmorph.
    :param flow_path: (default = '') the path to a voxelmorph transformation field that will be used to transform all data instead of
    predicting a new transformation if exist_transform is True.
    :param coords_input_file: (default = '') The path to a file with DeepLabCut coordinates based on which a DeepLabCut transformation
    should be carried out.
    :param atlas_label_list: (default = []) A list of aligned atlases in which each brain region is filled with a unique numeric label.
    This allows for consistent identification of brain regions across images. If original_label is True, this is an
    empty list.
    :param model: (default = 'models/unet_bundary.hdf5') The location (within the MesoNet repository) of a U-net model to be used for finding the boundaries
    of the brain region (as the default model does), or (if you have a specially trained model for this purpose)
    segmenting the entire brain into regions without the need for atlas alignment. Only choose another model if you have
    another model that you would like to use for segmenting the brain.
    :param region_labels: (default = False) If True, MesoNet will attempt to label each brain region according to the Allen Institute's
    Mouse Brain Atlas. Otherwise, MesoNet will label each region with a number. Please note that this feature is
    experimental!
    :param steps_per_epoch: (default = 300) During U-Net training, the number of steps that the model will take per epoch. Defaults to
    300 steps per epoch.
    :param epochs: (default = 60) During U-Net training, the number of epochs for which the model will run. Defaults to 60 epochs (set
    lower for online learning, e.g. if augmenting existing model).
    :param rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, horizontal_flip, fill_mode:
    (defaults: rotation_range = 0.3, width_shift_range = 0.05, height_shift_range = 0.05, shear_range = 0.05, zoom_range = 0.05, horizontal_flip = True, fill_mode = 'nearest') Keras image augmentation parameters for U-Net model training. See https://keras.io/api/preprocessing/image/ for
    full documentation.

    :Example:

    config_project('path/to/input/dir', 'path/to/output/dir', 'test')

    """

    # git_repo_base = 'C:/Users/mind reader/Desktop/mesonet/mesonet'
    git_repo_base = find_git_repo()
    print(git_repo_base)
    config = join(git_repo_base, config)
    model = join(git_repo_base, model)
    filename = "mesonet_config.yaml"
    data = dict()

    if mode == "test":
        filename = "mesonet_test_config.yaml"
        num_images = len(glob.glob(os.path.join(input_dir, "*.png")))
        data = dict(
            config=config,
            input_file=input_dir,
            output=output_dir,
            atlas=atlas,
            sensory_match=sensory_match,
            sensory_path=sensory_path,
            mat_save=mat_save,
            threshold=threshold,
            mask_generate=mask_generate,
            num_images=num_images,
            model=model,
            git_repo_base=git_repo_base,
            region_labels=region_labels,
            landmark_arr=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            use_unet=use_unet,
            use_dlc=use_dlc,
            atlas_to_brain_align=atlas_to_brain_align,
            olfactory_check=olfactory_check,
            plot_landmarks=plot_landmarks,
            align_once=align_once,
            atlas_label_list=atlas_label_list,
            original_label=original_label,
            use_voxelmorph=use_voxelmorph,
            exist_transform=exist_transform,
            voxelmorph_model=voxelmorph_model,
            template_path=template_path,
            flow_path=flow_path,
            coords_input_file=coords_input_file
        )
    elif mode == "train":
        filename = "mesonet_train_config.yaml"
        data = dict(
            input_file=input_dir,
            model_name=model_name,
            log_folder=output_dir,
            git_repo_base=git_repo_base,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            bodyparts=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode,
        )

    if glob.glob(os.path.join(input_dir, "*.mat")) or glob.glob(
        os.path.join(input_dir, "*.npy")
    ):
        convert_to_png(input_dir)

    with open(os.path.join(output_dir, filename), "w") as outfile:
        yaml.dump(data, outfile)

    config_file = os.path.join(output_dir, filename)
    return config_file


def parse_yaml(config_file):
    """
    Parses the config file and returns a dictionary with its parameters.

    :param config_file: The full path to a MesoNet config file (generated using mesonet.config_project())
    """
    with open(config_file, "r") as stream:
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
    _nsre = re.compile("([0-9]+)")
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
    ]


def find_git_repo():
    """
    Attempts to locate the MesoNet git repository on the computer.

    :return: The path to the MesoNet git repository.
    """
    # Preferred (faster) option to find mesonet git repo is to set it as an environment variable
    git_repo_base = ""
    try:
        git_repo_base = os.path.join(os.getenv("MESONET_GIT"), 'mesonet')
        print('MesoNet git repo found at {}, skipping directory check...'.format(git_repo_base))
    except:
        # If we can't find the environment variable, search for the mesonet git repository
        # solution to find git repository on computer adapted from:
        # https://stackoverflow.com/questions/5153317/python-how-to-do-a-system-wide-search-for-a-file-when-just-the-filename-not-pa

        # Check if we're using Google Colab; if we are, change the root directory to /content
        # solution adapted from:
        # https://stackoverflow.com/questions/53581278/test-if-notebook-is-running-on-google-colab
        in_colab = False
        try:
            import google.colab

            in_colab = True
        except:
            in_colab = False

        root_folder = "C:\\"
        git_repo_marker = "mesonet.txt"
        if platform == "linux" or platform == "linux2":
            # linux
            root_folder = "/home"
        elif platform == "darwin":
            # mac
            root_folder = "/Users"
        elif platform == "win32":
            # Windows
            # Get letter of current drive
            drive_letter = pathlib.Path.home().drive
            root_folder = "{}\\".format(drive_letter)
        elif in_colab:
            root_folder = "/content"
        for root, dirs, files in os.walk(root_folder):
            if git_repo_marker in files:
                git_repo_base = root
                break
    return git_repo_base


def convert_to_png(input_folder):
    """
    Utility for converting files from .mat or .npy formats (single or stacked) to .png images.

    :param input_folder: (required) A folder containing a .mat or .npy array (single or multi-dimensional) to be
    converted into an image (or images).
    """
    if glob.glob(os.path.join(input_folder, "*.mat")):
        input_file = glob.glob(os.path.join(input_folder, "*.mat"))[0]
        base_name = os.path.basename(input_file).split(".")[0]
        img_path = os.path.join(os.path.split(input_file)[0], base_name + ".png")
        print(img_path)
        mat = sio.loadmat(input_file)
        mat_shape = mat[list(mat.keys())[3]]
        if len(mat_shape.shape) > 2:
            for idx_arr in range(0, mat_shape.shape[2]):
                mat_layer = mat_shape[:, :, idx_arr]
                base_name_multi_idx = str(idx_arr) + "_" + base_name
                img_path_multi_idx = os.path.join(
                    os.path.split(input_file)[0], base_name_multi_idx + ".png"
                )
                cv2.imwrite(img_path_multi_idx, mat_layer)
        else:
            mat = mat[
                str(list({k: v for (k, v) in mat.items() if "__" not in k}.keys())[0])
            ]
            mat = mat * 255
            cv2.imwrite(img_path, mat)
        print(".mat written to .png!")
    elif glob.glob(os.path.join(input_folder, "*.npy")):
        input_file = glob.glob(os.path.join(input_folder, "*.npy"))[0]
        base_name = os.path.basename(input_file).split(".")[0]
        img_path = os.path.join(os.path.split(input_file)[0], base_name + ".png")
        npy = np.load(input_file)
        if npy.ndim == 3:
            for idx_arr, arr in enumerate(npy):
                base_name_multi_idx = str(idx_arr) + "_" + base_name
                img_path_multi_idx = os.path.join(
                    os.path.split(input_file)[0], base_name_multi_idx + ".png"
                )
                cv2.imwrite(img_path_multi_idx, arr * 255)
        else:
            npy = npy * 255
            cv2.imwrite(img_path, npy)


def plot_flow(flow_dir, output_dir):
    """
    Plots a flow field from VoxelMorph transformation flow files.

    :param flow_dir: (required) A directory containing one or more .npy flow files generated by a VoxelMorph alignment.
    :param output_dir: (required) The directory to which to save the flow files.
    :return:
    """
    flow_files = glob.glob(os.path.join(flow_dir, "*.npy"))
    if flow_files:
        for flow_idx, flow_file in enumerate(flow_files):
            print('Reading flow file {}'.format(flow_idx))
            flow_np = np.load(flow_file)
            ne.plot.flow([flow_np.squeeze()], width=5, show=False)
            plt.savefig(os.path.join(output_dir, '{}_flow_img.png'.format(flow_idx)))
    else:
        print("No .npy flow files found in current directory!")
