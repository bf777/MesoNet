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


def predictRegion(
    input_file,
    num_images,
    model,
    output,
    mat_save,
    threshold,
    mask_generate,
    git_repo_base,
    atlas_to_brain_align,
    dlc_pts,
    atlas_pts,
    olfactory_check,
    use_unet,
    plot_landmarks,
    atlas_label_list,
    align_once,
    region_labels,
    original_label,
):
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
    :param atlas_to_brain_align: If True, warp and register an atlas to the brain image; if False, warp and register a
    brain image to the atlas.
    :param dlc_pts: The landmarks for brain-atlas registration as determined by the DeepLabCut model.
    :param atlas_pts: The landmarks for brain-atlas registration from the original brain atlas.
    :param region_labels: Choose whether or not to attempt to label each region with its name from the Allen Institute
    Mouse Brain Atlas.
    :param olfactory_check: If True, draws olfactory bulb contours on the brain image.
    :param use_unet: Choose whether or not to identify the borders of the cortex using a U-net model.
    :param atlas_to_brain_align: If True, registers the atlas to each brain image. If False, registers each brain image
    to the atlas.
    :param plot_landmarks: If True, plots DeepLabCut landmarks (large circles) and original alignment landmarks (small
    circles) on final brain image.
    :param atlas_label_list: A list of aligned atlases in which each brain region is filled with a unique numeric label.
    This allows for consistent identification of brain regions across images. If original_label is True, this is an
    empty list.
    :param align_once: if True, carries out all alignments based on the alignment of the first atlas and brain. This can
    save time if you have many frames of the same brain with a fixed camera position.
    :param region_labels: choose whether to assign a name to each region based on an existing brain atlas (not currently
    implemented).
    :param original_label: if True, uses a brain region labelling approach that attempts to automatically sort brain
    regions in a consistent order (left to right by hemisphere, then top to bottom for vertically aligned regions). This
    approach may be more flexible if you're using a custom brain atlas (i.e. not one in which region is filled with a
    unique number).
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
    if not mask_generate:
        model_path = os.path.join(git_repo_base, "models", model)
        model_to_use = load_model(model_path)
    else:
        model_to_use = load_model(model)
    # Resizes and prepares images for prediction
    print(input_file)
    test_gen = testGenerator(
        input_file,
        output_mask_path,
        num_images,
        atlas_to_brain_align=atlas_to_brain_align,
    )
    # Makes predictions on each image
    results = model_to_use.predict_generator(test_gen, num_images, verbose=1)
    # Saves output mask
    saveResult(output_mask_path, results)
    if not mask_generate:
        plot_landmarks = False
        use_dlc = False
        # Predicts and identifies brain regions based on output mask
        applyMask(
            input_file,
            output_mask_path,
            output_overlay_path,
            output_overlay_path,
            mat_save,
            threshold,
            git_repo_base,
            atlas_to_brain_align,
            model,
            dlc_pts,
            atlas_pts,
            olfactory_check,
            use_unet,
            use_dlc,
            plot_landmarks,
            align_once,
            atlas_label_list,
            region_labels,
            original_label,
        )


def predict_regions(config_file):
    """
    Loads parameters into predictRegion from config file.
    :param config_file: The full path to a MesoNet config file (generated using mesonet.config_project())
    """
    cwd = os.getcwd()
    cfg = parse_yaml(config_file)
    input_file = cfg["input_file"]
    num_images = cfg["num_images"]
    model = os.path.join(cwd, cfg["model"])
    output = cfg["output"]
    mat_save = cfg["mat_save"]
    threshold = cfg["threshold"]
    mask_generate = True
    git_repo_base = cfg["git_repo_base"]
    region_labels = cfg["region_labels"]
    atlas_to_brain_align = cfg["atlas_to_brain_align"]
    dlc_pts = []
    atlas_pts = []
    olfactory_check = cfg["olfactory_check"]
    use_unet = cfg["use_unet"]
    plot_landmarks = cfg["plot_landmarks"]
    align_once = cfg["align_once"]
    atlas_label_list = cfg["atlas_label_list"]
    original_label = cfg["original_label"]

    predictRegion(
        input_file,
        num_images,
        model,
        output,
        mat_save,
        threshold,
        mask_generate,
        git_repo_base,
        atlas_to_brain_align,
        dlc_pts,
        atlas_pts,
        olfactory_check,
        use_unet,
        plot_landmarks,
        align_once,
        atlas_label_list,
        region_labels,
        original_label,
    )
