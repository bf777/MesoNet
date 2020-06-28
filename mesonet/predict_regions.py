"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
from mesonet.model import *
from mesonet.mask_functions import *
import os
from mesonet.utils import parse_yaml


def predictRegion(input_file, num_images, model, output, mat_save, threshold, mask_generate, git_repo_base,
                  atlas_to_brain_align, region_labels):
    """
    Segment brain images to predict the location of brain regions.
    :param input_file: Input folder containing brain images.
    :param num_images: Number of brain images to be analyzed.
    :param model: Prediction model (.hdf5 file) to be used.
    :param output: Overall output folder into which all files will be saved.
    :param mat_save: Choose whether or not to save each brain region contour and centre as a .mat file (for MATLAB).
    :param threshold: Threshold for segmentation algorithm.
    :param mask_generate: Choose whether or not to only generate masks of the brain contour from this function.
    :param git_repo_base: The path to the base git repository containing necessary resources for MesoNet (reference
    atlases, DeepLabCut config files, etc.)
    :param region_labels: Choose whether or not to attempt to label each region with its name from the Allen Institute
    Mouse Brain Atlas.
    """
    # Create and define save folders for each output of the prediction
    # Output folder for basic mask (used later in prediction)
    output_mask_path = os.path.join(output, "output_mask")
    # Output folder for transparent masks and masks overlaid onto brain image
    output_overlay_path = os.path.join(output, "output_overlay")
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    if not os.path.isdir(output_overlay_path):
        os.mkdir(output_overlay_path)
    # Loads in existing model
    print(model)
    model = load_model(model)
    # Resizes and prepares images for prediction
    test_gen = testGenerator(input_file, num_images)
    # Makes predictions on each image
    results = model.predict_generator(test_gen, num_images, verbose=1)
    # Saves output mask
    saveResult(output_mask_path, results)
    if not mask_generate:
        # Predicts and identifies brain regions based on output mask
        applyMask(input_file, output_mask_path, output_overlay_path, output, mat_save, threshold, git_repo_base,
                  atlas_to_brain_align, region_labels)


def predict_regions(config_file):
    """
    Loads parameters into predictRegion from config file.
    :param config_file: The full path to a MesoNet config file (generated using mesonet.config_project())
    """
    cwd = os.getcwd()
    cfg = parse_yaml(config_file)
    input_file = cfg['input_file']
    num_images = cfg['num_images']
    model = os.path.join(cwd, cfg['model'])
    output = cfg['output']
    mat_save = cfg['mat_save']
    threshold = cfg['threshold']
    mask_generate = True
    git_repo_base = cfg['git_repo_base']
    region_labels = cfg['region_labels']
    atlas_to_brain_align = cfg['atlas_to_brain_align']
    predictRegion(input_file, num_images, model, output, mat_save, threshold, mask_generate, git_repo_base,
                  atlas_to_brain_align, region_labels)
