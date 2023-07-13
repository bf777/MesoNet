"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the Creative Commons Attribution 4.0 International License (see LICENSE for details)
"""
import deeplabcut
from mesonet.atlas_brain_matching import atlasBrainMatch
from mesonet.utils import parse_yaml, natural_sort_key
from deeplabcut.utils.auxiliaryfunctions import read_config, write_config
import cv2
import glob
import imageio
import os
import numpy as np
import re
from sys import platform


def DLCPredict(
    config,
    input_file,
    output,
    atlas,
    sensory_match,
    sensory_path,
    mat_save,
    threshold,
    git_repo_base,
    region_labels,
    landmark_arr,
    use_unet,
    use_dlc,
    atlas_to_brain_align,
    model,
    olfactory_check,
    plot_landmarks,
    align_once,
    original_label,
    use_voxelmorph,
    exist_transform,
    voxelmorph_model,
    template_path,
    flow_path,
    coords_input_file,
):
    """
    Takes a directory of brain images and predicts cortical landmark locations (left and right suture, bregma, and
    lambda) using a DeepLabCut model.

    :param config: Select the config file for the DeepLabCut model to be used for landmark estimation.
    :param input_file: The folder containing the brain images to be analyzed.
    :param output: The folder to which we save the output brain image, labelled with the predicted locations of each
    landmark.
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
    :param git_repo_base: The path to the base git repository containing necessary resources for MesoNet (reference
    atlases, DeepLabCut config files, etc.)
    :param region_labels: Choose whether or not to attempt to label each region with its name from the Allen Institute
    Mouse Brain Atlas.
    :param landmark_arr: A list of numbers indicating which landmarks should be used by the model.
    :param use_unet: Choose whether or not to define the borders of the cortex using a U-net model.
    :param use_dlc: Choose whether or not to try and register the atlas and brain image using a DeepLabCut model.
    :param atlas_to_brain_align: If True, registers the atlas to each brain image. If False, registers each brain image
    to the atlas.
    :param model: The name of the U-net model (for passthrough to mask_functions.py)
    :param olfactory_check: If True, draws olfactory bulb contours on the brain image.
    :param plot_landmarks: If True, plots DeepLabCut landmarks (large circles) and original alignment landmarks (small
    circles) on final brain image.
    :param align_once: if True, carries out all alignments based on the alignment of the first atlas and brain. This can
    save time if you have many frames of the same brain with a fixed camera position.
    :param original_label: if True, uses a brain region labelling approach that attempts to automatically sort brain
    regions in a consistent order (left to right by hemisphere, then top to bottom for vertically aligned regions). This
    approach may be more flexible if you're using a custom brain atlas (i.e. not one in which region is filled with a
    unique number).
    :param use_voxelmorph: Choose whether or not to apply a local deformation registration for image registration,
    using a voxelmorph model.
    :param exist_transform: if True, uses an existing voxelmorph transformation field for all data instead of predicting
    a new transformation.
    :param voxelmorph_model: the name of a .h5 model located in the models folder of the git repository for MesoNet,
    generated using voxelmorph and containing weights for a voxelmorph local deformation model.
    :param template_path: the path to a template atlas (.npy or .mat) to which the brain image will be aligned in
    voxelmorph.
    :param flow_path: the path to a voxelmorph transformation field that will be used to transform all data instead of
    predicting a new transformation if exist_transform is True.
    """
    img_array = []
    if sensory_match == 1:
        sensory_match = True
    else:
        sensory_match = False
    if sensory_match:
        sensory_img_dir = sensory_path
    else:
        sensory_img_dir = ""
    tif_list = glob.glob(os.path.join(input_file, "*tif"))
    if tif_list:
        print(tif_list)
        tif_stack = imageio.mimread(os.path.join(input_file, tif_list[0]))
        filenames = tif_stack
    else:
        filenames = glob.glob(os.path.join(input_file, "*.png"))
        filenames.sort(key=natural_sort_key)

    size = (512, 512)
    print(len(filenames))
    for filename in filenames:
        print(filename)
        if tif_list:
            img = filename
            img = np.uint8(img)
            img = cv2.resize(img, size)
            height, width = img.shape
        else:
            img = cv2.imread(filename)
            img = np.uint8(img)
            img = cv2.resize(img, size)
            height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if len(img_array) > 0:
        video_output_path = os.path.join(output, "dlc_output")

        video_name = os.path.join(video_output_path, "tmp_video.mp4")

        if not os.path.isdir(video_output_path):
            os.mkdir(video_output_path)
        if not coords_input_file:
            # fourcc = cv2.VideoWriter_fourcc(*'DIB ')
            if platform == "linux" or platform == "linux2" or platform == "darwin":
                fourcc = cv2.VideoWriter_fourcc("M", "P", "E", "G")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_name, fourcc, 30, size)
            for i in img_array:
                # print("img {} written!".format(i))
                out.write(i)
            out.release()

            print("DLC config file path: {}".format(config))

            deeplabcut.analyze_videos(
                config, [video_output_path], videotype=".mp4", save_as_csv=True
            )
            deeplabcut.create_labeled_video(config, [video_name], filtered=True)
            #             if "2." in deeplabcut.__version__:
            #                 scorer_name = "DeepCut"
            #             else:
            scorer_name = "DLC"
            output_video_name = ""
            coords_input = ""
            for filename in glob.glob(
                os.path.join(video_output_path, "tmp_video" + scorer_name + "*.*")
            ):
                try:
                    if ".mp4" in filename:
                        output_video_name = filename
                    elif ".csv" in filename:
                        coords_input = filename
                except FileNotFoundError:
                    print(
                        "Please ensure that an output video and corresponding datafile from DeepLabCut are in the folder!"
                    )

            cap = cv2.VideoCapture(output_video_name)
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(
                    os.path.join(video_output_path, "{}.png".format(str(i))), frame
                )
                i += 1
        else:
            coords_input = coords_input_file

        os.chdir(video_output_path)

        print("Landmark prediction complete!")
        if not atlas:
            atlasBrainMatch(
                input_file,
                sensory_img_dir,
                coords_input,
                sensory_match,
                mat_save,
                threshold,
                git_repo_base,
                region_labels,
                landmark_arr,
                use_unet,
                use_dlc,
                atlas_to_brain_align,
                model,
                olfactory_check,
                plot_landmarks,
                align_once,
                original_label,
                use_voxelmorph,
                exist_transform,
                voxelmorph_model,
                template_path,
                flow_path,
            )


def DLCPredictBehavior(config, input_file, output):
    """
    Takes a video of animal behaviour and predicts body movements based on a supplied DeepLabCut body movement model.

    :param config: The path to the DeepLabCut configuration file.
    :param input_file: The folder containing the behavioural images to be analyzed.
    :param output: The folder to which we save the predicted body movements.
    """
    video_array = []
    img_array = []
    print(input_file)
    for filename in glob.glob(os.path.join(input_file, "*.mp4")):
        video_array.append(os.path.join(input_file, filename))

    for s in glob.glob(os.path.join(input_file, "*.png")):
        _nsre = re.compile("([0-9]+)")
        return [
            int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
        ]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if len(img_array) > 0:
        video_output_path = os.path.join(output, "dlc_output", "behavior")
        video_name = os.path.join(video_output_path, "behavior_video.mp4")
        video_output_path = [video_output_path]
        video_name = [video_name]

        if not os.path.isdir(video_output_path[0]):
            os.mkdir(video_output_path[0])
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MP4V"), 30, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    elif len(video_array) > 0:
        video_output_path = video_array
        video_name = video_array

    deeplabcut.analyze_videos(
        config, video_output_path, videotype=".mp4", save_as_csv=True, destfolder=output
    )
    deeplabcut.create_labeled_video(
        config, video_name, filtered=True, destfolder=output
    )
    cv2.destroyAllWindows()


def DLCPrep(project_name, your_name, img_path, output_dir_base, copy_videos_bool=True):
    img_array = []
    filenames = glob.glob(os.path.join(img_path, "*.png"))
    filenames.sort(key=natural_sort_key)
    size = (512, 512)
    for filename in filenames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if len(img_array) > 0:
        if not os.path.isdir(os.path.join(output_dir_base, "img_for_label")):
            os.mkdir(os.path.join(output_dir_base, "img_for_label"))
        video_output_path = os.path.join(output_dir_base, "img_for_label")
        video_name = os.path.join(video_output_path, "video_for_label.mp4")

        if not os.path.isdir(video_output_path):
            os.mkdir(video_output_path)
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MP4V"), 30, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        config_path = deeplabcut.create_new_project(
            project_name,
            your_name,
            [video_name],
            copy_videos=copy_videos_bool,
            working_directory=output_dir_base,
        )
        return config_path


def DLCLabel(config_path):
    """
    Utility to facilitate labelling DeepLabCut frames from the MesoNet GUI.

    :param config_path: (required) The path to a DeepLabCut configuration file (.yaml).
    """
    deeplabcut.extract_frames(config_path, crop=False)
    deeplabcut.label_frames(config_path)
    deeplabcut.check_labels(config_path)


def DLCTrain(config_path, displayiters, saveiters, maxiters):
    """
    Utility to facilitate training a DeepLabCut model from the MesoNet GUI.

    :param config_path: (required) The path to a DeepLabCut configuration file (.yaml).
    :param displayiters: (required) The interval between which intervals should be shown.
    :param saveiters: (required) The interval after which the model should be saved.
    :param maxiters: (required) The number of iterations after which the model should finish training.
    """
    deeplabcut.create_training_dataset(config_path)
    deeplabcut.train_network(
        config_path, displayiters=displayiters, saveiters=saveiters, maxiters=maxiters
    )


def DLC_edit_bodyparts(config_path, new_bodyparts):
    """
    Utility to facilitate changing the names and number of bodyparts in a DeepLabCut model from the MesoNet GUI.

    :param config_path: (required) The path to a DeepLabCut configuration file (.yaml).
    :param new_bodyparts: (required) A list of new bodypart names to write
    """
    dlc_cfg = read_config(config_path)
    dlc_cfg["bodyparts"] = new_bodyparts
    write_config(config_path, dlc_cfg)


def predict_dlc(config_file):
    """
    Loads parameters into DLCPredict from config file.

    :param config_file: The full path to a MesoNet config file (generated using mesonet.config_project())
    """
    cwd = os.getcwd()
    cfg = parse_yaml(config_file)
    config = cfg["config"]
    atlas = cfg["atlas"]
    sensory_match = cfg["sensory_match"]
    sensory_path = cfg["sensory_path"]
    input_file = cfg["input_file"]
    output = cfg["output"]
    mat_save = cfg["mat_save"]
    threshold = cfg["threshold"]
    git_repo_base = cfg["git_repo_base"]
    region_labels = cfg["region_labels"]
    landmark_arr = cfg["landmark_arr"]
    use_unet = cfg["use_unet"]
    use_dlc = cfg["use_dlc"]
    atlas_to_brain_align = cfg["atlas_to_brain_align"]
    model = os.path.join(cwd, cfg["model"])
    olfactory_check = cfg["olfactory_check"]
    plot_landmarks = cfg["plot_landmarks"]
    align_once = cfg["align_once"]
    original_label = cfg["original_label"]
    use_voxelmorph = cfg["use_voxelmorph"]
    exist_transform = cfg["exist_transform"]
    voxelmorph_model = cfg["voxelmorph_model"]
    template_path = cfg["template_path"]
    flow_path = cfg["flow_path"]
    coords_input_file = cfg["coords_input_file"]
    DLCPredict(
        config,
        input_file,
        output,
        atlas,
        sensory_match,
        sensory_path,
        mat_save,
        threshold,
        git_repo_base,
        region_labels,
        landmark_arr,
        use_unet,
        use_dlc,
        atlas_to_brain_align,
        model,
        olfactory_check,
        plot_landmarks,
        align_once,
        original_label,
        use_voxelmorph,
        exist_transform,
        voxelmorph_model,
        template_path,
        flow_path,
        coords_input_file,
    )
