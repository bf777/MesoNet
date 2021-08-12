"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
This file has been adapted from data.py in https://github.com/zhixuhao/unet
"""
from mesonet.utils import natural_sort_key
import numpy as np
import scipy.io as sio
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage.util import img_as_ubyte
import cv2
import imageio
import imutils
import scipy
import pylab
from PIL import Image
import pandas as pd
from keras import backend as k
from polylabel import polylabel

# Set background colour as black to fix issue with more than one background region being identified.
Background = [0, 0, 0]
# Foreground (cortex) should be rendered as white.
Region = [255, 255, 255]

COLOR_DICT = np.array([Background, Region])
NUM_COLORS = 9


def testGenerator(
    test_path,
    output_mask_path,
    num_image=60,
    target_size=(512, 512),
    flag_multi_class=False,
    as_gray=True,
    atlas_to_brain_align=True,
):
    """
    Import images and resize it to the target size of the model.
    :param test_path: path to input images
    :param num_image: number of input images
    :param target_size: target image size as defined in the Keras model
    :param flag_multi_class: flag the input images as having multiple classes
    :param as_gray: If input image is grayscale, process data input accordingly
    """
    suff = "png"
    img_list = glob.glob(os.path.join(test_path, "*png"))
    img_list.sort(key=natural_sort_key)
    tif_list = glob.glob(os.path.join(test_path, "*tif"))
    if tif_list:
        tif_stack = imageio.mimread(os.path.join(test_path, tif_list[0]))
        num_image = len(tif_stack)
    for i in range(num_image):
        if len(tif_list) > 0:
            print("TIF detected")
            img = tif_stack[i]
            img = np.uint8(img)
            img = cv2.resize(img, target_size)
        elif len(tif_list) == 0:
            if atlas_to_brain_align:
                img = io.imread(os.path.join(test_path, img_list[i]))
            else:
                try:
                    img = io.imread(
                        os.path.join(test_path, "{}_brain_warp.{}".format(i, suff))
                    )
                except:
                    img = io.imread(os.path.join(test_path, img_list[i]))
            img = trans.resize(img, target_size)
        img = img_as_ubyte(img)
        io.imsave(os.path.join(output_mask_path, "{}.{}".format(i, suff)), img)
        img = io.imread(
            os.path.join(output_mask_path, "{}.{}".format(i, suff)), as_gray=as_gray
        )
        if img.dtype == "uint8":
            img = img / 255
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


def labelVisualize(num_class, color_dict, img):
    """
    Visualize labels on image based on input classes.
    :param num_class: number of classes
    :param color_dict: dictionary of colours defined at top of file
    :param img: input image
    """
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    """
    Saves the predicted mask from each brain image to the save folder.
    :param save_path: path to overall folder for saving images
    :param npyfile: results file output after model has made predictions
    :param flag_multi_class: flag the output images as having multiple classes
    :param num_class: If flag_multi_class is True, decide how many classes to categorize each brain image into.
    """
    for i, item in enumerate(npyfile):
        img = (
            labelVisualize(num_class, COLOR_DICT, item)
            if flag_multi_class
            else item[:, :, 0]
        )
        io.imsave(os.path.join(save_path, "{}.png".format(i)), img)


def returnResult(save_path, npyfile):
    """
    Saves the predicted mask from each brain image to the save folder.
    :param save_path: path to overall folder for saving images
    :param npyfile: results file output after model has made predictions
    :param flag_multi_class: flag the output images as having multiple classes
    :param num_class: If flag_multi_class is True, decide how many classes to categorize each brain image into.
    """
    img = npyfile[0][:, :, 0]
    return img


def atlas_to_mask(
    atlas_path,
    mask_input_path,
    mask_warped_path,
    mask_output_path,
    n,
    use_unet,
    use_voxelmorph,
    atlas_to_brain_align,
    git_repo_base,
    olfactory_check,
    olfactory_bulbs_to_use,
    atlas_label
):
    """
    Overlays the U-net mask and a smoothing mask for the cortical boundaries on the transformed brain atlas.
    :param atlas_path: The path to the atlas to be transformed
    :param mask_input_path: The path to the U-net mask corresponding to the input atlas
    :param mask_warped_path: The path to a mask transformed alongside the atlas to correct for gaps between the U-net
    cortical boundaries and the brain atlas.
    :param mask_output_path: The output path of the completed atlas with overlaid masks
    :param n: The number of the current atlas and corresponding transformed mask
    :param use_unet: Choose whether or not to define the borders of the cortex using a U-net model.
    :param atlas_to_brain_align: If True, registers the atlas to each brain image. If False, registers each brain image
    to the atlas.
    :param git_repo_base: The path to the base git repository containing necessary resources for MesoNet (reference
    atlases, DeepLabCut config files, etc.)
    :param olfactory_check: If True, draws olfactory bulb contours on the brain image.
    :param atlas_label: An atlas in which each brain region is filled with a unique numeric label.
    This allows for consistent identification of brain regions across images. If original_label is True, this is an
    empty list.
    """
    atlas = cv2.imread(atlas_path, cv2.IMREAD_GRAYSCALE)
    mask_warped = cv2.imread(mask_warped_path, cv2.IMREAD_GRAYSCALE)
    if use_unet:
        mask_input = cv2.imread(mask_input_path, cv2.IMREAD_GRAYSCALE)
        mask_input_orig = mask_input
        if olfactory_check and not use_voxelmorph:
            cnts_for_olfactory = cv2.findContours(
                mask_input.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cnts_for_olfactory = imutils.grab_contours(cnts_for_olfactory)
            if len(cnts_for_olfactory) == 3:
                olfactory_bulbs = sorted(
                    cnts_for_olfactory, key=cv2.contourArea, reverse=True
                )[1:3]
            else:
                olfactory_bulbs = sorted(
                    cnts_for_olfactory, key=cv2.contourArea, reverse=True
                )[2:4]
        io.imsave(os.path.join(mask_output_path, "{}_mask.png".format(n)), mask_input)
        # Adds the common white regions of the atlas and U-net mask together into a binary image.
        if atlas_to_brain_align:
            # FOR ALIGNING ATLAS TO BRAIN
            if use_voxelmorph and olfactory_check:
                olfactory_bulbs_to_add = olfactory_bulbs_to_use
                mask_input = cv2.bitwise_and(atlas, mask_warped)
                mask_input_orig = cv2.bitwise_and(mask_input, mask_warped)
            else:
                if olfactory_check:
                    olfactory_bulbs_to_add = olfactory_bulbs
                mask_input_orig = cv2.bitwise_and(mask_input, mask_warped)
                mask_input = cv2.bitwise_and(atlas, mask_input)
                mask_input = cv2.bitwise_and(mask_input, mask_warped)
            if len(atlas_label) > 0:
                atlas_label[np.where(mask_input == 0)] = 1000
            if olfactory_check:
                for bulb in olfactory_bulbs_to_add:
                    cv2.fillPoly(mask_input, pts=[bulb], color=[255, 255, 255])
                    cv2.fillPoly(mask_input_orig, pts=[bulb], color=[255, 255, 255])
                if len(atlas_label) > 0:
                    try:
                        cv2.fillPoly(atlas_label, pts=[olfactory_bulbs_to_add[0]], color=[300])
                        cv2.fillPoly(atlas_label, pts=[olfactory_bulbs_to_add[1]], color=[400])
                        atlas_label[np.where(atlas_label == 300)] = 300
                        atlas_label[np.where(atlas_label == 400)] = 400
                    except:
                        print('No olfactory bulb found!')
                        # If olfactory bulbs that match with labelling atlas
                        # can't be found, regenerate the original atlas
                        mask_input = cv2.imread(mask_input_path, cv2.IMREAD_GRAYSCALE)
                        mask_input_orig = mask_input
                        mask_input_orig = cv2.bitwise_and(mask_input, mask_warped)
                        mask_input = cv2.bitwise_and(atlas, mask_input)
                        mask_input = cv2.bitwise_and(mask_input, mask_warped)
        else:
            # FOR ALIGNING BRAIN TO ATLAS
            mask_input = cv2.bitwise_and(atlas, mask_warped)
            mask_input_orig = cv2.bitwise_and(mask_input, mask_warped)
            if olfactory_check and len(olfactory_bulbs_to_use) > 0:
                for bulb in olfactory_bulbs_to_use:
                    cv2.fillPoly(mask_input, pts=[bulb], color=[255, 255, 255])
                    cv2.fillPoly(mask_input_orig, pts=[bulb], color=[255, 255, 255])
                if len(atlas_label) > 0:
                    try:
                        cv2.fillPoly(atlas_label, pts=[olfactory_bulbs_to_use[0]], color=[300])
                        cv2.fillPoly(atlas_label, pts=[olfactory_bulbs_to_use[1]], color=[400])
                        atlas_label[np.where(atlas_label == 300)] = 300
                        atlas_label[np.where(atlas_label == 400)] = 400
                    except:
                        print('No olfactory bulb found!')
                        # If olfactory bulbs that match with labelling atlas
                        # can't be found, regenerate the original atlas
                        mask_input = cv2.imread(mask_input_path, cv2.IMREAD_GRAYSCALE)
                        mask_input_orig = mask_input
                        mask_input = cv2.bitwise_and(atlas, mask_warped)
                        mask_input_orig = cv2.bitwise_and(mask_input, mask_warped)
        io.imsave(os.path.join(mask_output_path, "{}_mask_no_atlas.png".format(n)), mask_input_orig)
    else:
        mask_input = cv2.bitwise_and(atlas, mask_warped)
        if len(atlas_label) > 0:
            atlas_label[np.where(mask_input == 0)] = 1000
        if olfactory_check:
            olfactory_path = os.path.join(git_repo_base, "atlases")
            olfactory_left = cv2.imread(
                os.path.join(olfactory_path, "02.png"), cv2.IMREAD_GRAYSCALE
            )
            olfactory_right = cv2.imread(
                os.path.join(olfactory_path, "01.png"), cv2.IMREAD_GRAYSCALE
            )
            cnts_for_olfactory_left, hierarchy = cv2.findContours(
                olfactory_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )[-2:]
            olfactory_left_cnt = min(cnts_for_olfactory_left, key=cv2.contourArea)
            cnts_for_olfactory_right, hierarchy = cv2.findContours(
                olfactory_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )[-2:]
            olfactory_right_cnt = min(cnts_for_olfactory_right, key=cv2.contourArea)
            cv2.fillPoly(mask_input, pts=[olfactory_left_cnt], color=[255, 255, 255])
            cv2.fillPoly(mask_input, pts=[olfactory_right_cnt], color=[255, 255, 255])
    io.imsave(os.path.join(mask_output_path, "{}.png".format(n)), mask_input)
    return atlas_label


def inpaintMask(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    for cnt in cnts:
        cv2.fillPoly(mask, pts=[cnt], color=[255, 255, 255])
    return mask


def applyMask(
    image_path,
    mask_path,
    save_path,
    segmented_save_path,
    mat_save,
    threshold,
    git_repo_base,
    bregma_list,
    atlas_to_brain_align,
    model,
    dlc_pts,
    atlas_pts,
    olfactory_check,
    use_unet,
    use_dlc,
    use_voxelmorph,
    plot_landmarks,
    align_once,
    atlas_label_list,
    olfactory_bulbs_to_use_list,
    region_labels=True,
    original_label=False,
):
    """
    Use mask output from model to segment brain image into brain regions, and save various outputs.
    :param image_path: path to folder where brain images are saved
    :param mask_path: path to folder where masks are saved
    :param save_path: path to overall folder for saving all images
    :param segmented_save_path: path to overall folder for saving segmented/labelled brain images
    :param mat_save: choose whether or not to output brain regions to .mat files
    :param threshold: set threshold for segmentation of foregrounds
    :param git_repo_base: The path to the base git repository containing necessary resources for MesoNet (reference
    atlases, DeepLabCut config files, etc.)
    :param bregma_list: The list of bregma locations (or landmarks closest to bregma).
    :param region_labels: Choose whether or not to attempt to label each region with its name from the Allen Institute
    Mouse Brain Atlas.
    :param use_unet: Choose whether or not to define the borders of the cortex using a U-net model.
    :param atlas_to_brain_align: If True, registers the atlas to each brain image. If False, registers each brain image
    to the atlas.
    :param model: The name of the U-net model (for passthrough to mask_functions.py)
    :param dlc_pts: The landmarks for brain-atlas registration as determined by the DeepLabCut model.
    :param atlas_pts: The landmarks for brain-atlas registration from the original brain atlas.
    :param olfactory_check: If True, draws olfactory bulb contours on the brain image.
    :param plot_landmarks: If True, plots DeepLabCut landmarks (large circles) and original alignment landmarks (small
    circles) on final brain image.
    :param atlas_label_list: A list of aligned atlases in which each brain region is filled with a unique numeric label.
    This allows for consistent identification of brain regions across images. If original_label is True, this is an
    empty list.
    :param align_once: If True, carries out all alignments based on the alignment of the first atlas and brain. This can
    save time if you have many frames of the same brain with a fixed camera position.
    :param region_labels: choose whether to assign a name to each region based on an existing brain atlas (not currently
    implemented).
    :param original_label: If True, uses a brain region labelling approach that attempts to automatically sort brain
    regions in a consistent order (left to right by hemisphere, then top to bottom for vertically aligned regions). This
    approach may be more flexible if you're using a custom brain atlas (i.e. not one in which each region is filled with
    a unique number).
    """

    tif_list = glob.glob(os.path.join(image_path, "*tif"))
    if atlas_to_brain_align:
        if use_dlc and align_once:
            image_name_arr = glob.glob(os.path.join(mask_path, "*_brain_warp.png"))
        else:
            image_name_arr = glob.glob(os.path.join(image_path, "*.png"))
        image_name_arr.sort(key=natural_sort_key)
        if tif_list:
            tif_stack = imageio.mimread(os.path.join(image_path, tif_list[0]))
            image_name_arr = tif_stack
    else:
        # FOR ALIGNING BRAIN TO ATLAS
        image_name_arr = glob.glob(os.path.join(mask_path, "*_brain_warp.png"))
        image_name_arr.sort(key=natural_sort_key)

    region_bgr_lower = (220, 220, 220) # 220
    region_bgr_upper = (255, 255, 255)
    base_c_max = []
    count = 0
    # Find the contours of an existing set of brain regions (to be used to identify each new brain region by shape)
    mat_files = glob.glob(os.path.join(git_repo_base, "atlases/mat_contour_base/*.mat"))
    mat_files.sort(key=natural_sort_key)

    # adapt. from https://stackoverflow.com/questions/3016283/create-a-color-generator-from-given-colormap-in-matplotlib
    cm = pylab.get_cmap("viridis")
    colors = [cm(1.0 * i / NUM_COLORS)[0:3] for i in range(NUM_COLORS)]
    colors = [tuple(color_idx * 255 for color_idx in color_t) for color_t in colors]
    for file in mat_files:
        mat = scipy.io.loadmat(
            os.path.join(git_repo_base, "atlases/mat_contour_base/", file)
        )
        mat = mat["vect"]
        ret, thresh = cv2.threshold(mat, 5, 255, cv2.THRESH_BINARY)
        base_c = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        base_c = imutils.grab_contours(base_c)
        base_c_max.append(max(base_c, key=cv2.contourArea))
    # if not atlas_to_brain_align and use_unet:
    #     # FOR ALIGNING ATLAS TO BRAIN
    #     num_images = len(glob.glob(os.path.join(mask_path, "*_brain_warp*")))
    #     output = os.path.join(mask_path, "..")
    #     from mesonet.predict_regions import predictRegion
    #
    #     mask_generate = True
    #     tif_list = glob.glob(os.path.join(image_path, "*tif"))
    #     if tif_list:
    #         input_path = image_path
    #     else:
    #         input_path = mask_path
    #     predictRegion(
    #         input_path,
    #         num_images,
    #         model,
    #         output,
    #         mat_save,
    #         threshold,
    #         mask_generate,
    #         git_repo_base,
    #         atlas_to_brain_align,
    #         dlc_pts,
    #         atlas_pts,
    #         olfactory_check,
    #         use_unet,
    #         plot_landmarks,
    #         align_once,
    #         atlas_label_list,
    #         region_labels,
    #         original_label,
    #     )
    for i, item in enumerate(image_name_arr):
        label_num = 0
        if not atlas_to_brain_align:
            atlas_path = os.path.join(mask_path, "{}_atlas.png".format(str(i)))
            mask_input_path = os.path.join(mask_path, "{}.png".format(i))
            mask_warped_path = os.path.join(
                mask_path, "{}_mask_warped.png".format(str(i))
            )
            if olfactory_check:
                olfactory_bulbs_to_use = olfactory_bulbs_to_use_list[i]
            else:
                olfactory_bulbs_to_use = []
            atlas_to_mask(
                atlas_path,
                mask_input_path,
                mask_warped_path,
                mask_path,
                i,
                use_unet,
                use_voxelmorph,
                atlas_to_brain_align,
                git_repo_base,
                olfactory_check,
                olfactory_bulbs_to_use,
                []
            )
        new_data = []
        if len(tif_list) != 0 and atlas_to_brain_align:
            img = item
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(item)
        if atlas_to_brain_align:
            img = cv2.resize(img, (512, 512))
        if use_dlc:
            bregma_x, bregma_y = bregma_list[i]
        else:
            bregma_x, bregma_y = [round(img.shape[0]/2), round(img.shape[1]/2)]
            original_label = True
        if use_voxelmorph and i == 1:
            mask = cv2.imread(os.path.join(mask_path, "{}.png".format(0)))
        else:
            mask = cv2.imread(os.path.join(mask_path, "{}.png".format(i)))
        mask = cv2.resize(mask, (img.shape[0], img.shape[1]))
        # Get the region of the mask that is white
        mask_color = cv2.inRange(mask, region_bgr_lower, region_bgr_upper)
        io.imsave(os.path.join(save_path, "{}_mask_binary.png".format(i)), mask_color)
        # Marker labelling
        # noise removal
        kernel = np.ones((3, 3), np.uint8)  # 3, 3
        mask_color = np.uint8(mask_color)
        thresh_atlas, atlas_bw = cv2.threshold(mask_color, 128, 255, 0)
        # if atlas_to_brain_align and use_dlc:
        #    atlas_bw = cv2.dilate(atlas_bw, kernel, iterations=1)  # 1
        # io.imsave(os.path.join(save_path, "{}_atlas_binary.png".format(i)), atlas_bw)

        if not atlas_to_brain_align:
            watershed_run_rule = True
        else:
            if len(tif_list) == 0:
                watershed_run_rule = True
            else:
                watershed_run_rule = i == 0
        if align_once:
            watershed_run_rule = i == 0

        labels_from_region = []

        if watershed_run_rule:
            orig_list = []
            orig_list_labels = []
            orig_list_labels_left = []
            orig_list_labels_right = []
            # unique_regions = (np.unique(atlas_label)).tolist()
            # unique_regions = [e for e in unique_regions if e.is_integer()]
            unique_regions = [
                -275,
                -268,
                -255,
                -249,
                -164,
                -150,
                -143,
                -136,
                -129,
                -98,
                -78,
                -71,
                -64,
                -57,
                -50,
                -43,
                -36,
                -29,
                -21,
                -15,
                0,
                15,
                21,
                29,
                36,
                43,
                50,
                57,
                64,
                71,
                78,
                98,
                129,
                136,
                143,
                150,
                164,
                249,
                255,
                268,
                275,
                300,
                400,
            ]
            cnts_orig = []
            # Find contours in original aligned atlas
            if atlas_to_brain_align and not original_label:
                np.savetxt(
                    "atlas_label_list_{}.csv".format(i),
                    atlas_label_list[i],
                    delimiter=",",
                )
                for region_idx in unique_regions:
                    if region_idx in [300, 400]:
                        # workaround to address olfactory contours not being found
                        region = cv2.inRange(
                            atlas_label_list[i], region_idx - 5, region_idx + 5
                        )
                        cnt_for_idx, hierarchy = cv2.findContours(
                            region.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                        )[-2:]
                        if len(cnt_for_idx) >= 1:
                            cnt_for_idx = cnt_for_idx[0]
                    else:
                        region = cv2.inRange(
                            atlas_label_list[i], region_idx, region_idx
                        )
                        cnt_for_idx = cv2.findContours(
                            region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                        )
                        cnt_for_idx = imutils.grab_contours(cnt_for_idx)
                        if len(cnt_for_idx) >= 1:
                            cnt_for_idx = max(cnt_for_idx, key=cv2.contourArea)
                    if len(cnt_for_idx) >= 1:
                        cnts_orig.append(cnt_for_idx)
                        labels_from_region.append(region_idx)
            else:
                cnts_orig = cv2.findContours(
                    atlas_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                cnts_orig = imutils.grab_contours(cnts_orig)
            if not use_dlc:
                cnts_orig, hierarchy = cv2.findContours(
                    atlas_bw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )[-2:]
            labels_cnts = []
            for (num_label, cnt_orig) in enumerate(cnts_orig):
                labels_cnts.append(cnt_orig)
                try:
                    cv2.drawContours(img, cnt_orig, -1, (255, 0, 0), 1)
                except:
                    print("Could not draw contour!")
                # try:
                if atlas_to_brain_align:
                    c_orig_as_list = cnt_orig.tolist()
                    c_orig_as_list = [[c_val[0] for c_val in c_orig_as_list]]
                else:
                    c_orig_as_list = cnt_orig.tolist()
                    c_orig_as_list = [[c_val[0] for c_val in c_orig_as_list]]
                orig_polylabel = polylabel(c_orig_as_list)
                orig_x, orig_y = int(orig_polylabel[0]), int(orig_polylabel[1])

                if not original_label and atlas_to_brain_align:
                    label_to_use = unique_regions.index(labels_from_region[num_label])
                    (text_width, text_height) = cv2.getTextSize(
                        str(label_to_use), cv2.FONT_HERSHEY_SIMPLEX, 0.4, thickness=1
                    )[0]
                    label_jitter = 0
                    label_color = (0, 0, 255)
                    cv2.rectangle(
                        img,
                        (orig_x + label_jitter, orig_y + label_jitter),
                        (
                            orig_x + label_jitter + text_width,
                            orig_y + label_jitter - text_height,
                        ),
                        (255, 255, 255),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        img,
                        str(label_to_use),
                        (int(orig_x + label_jitter), int(orig_y + label_jitter)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        label_color,
                        1,
                    )
                    label_num += 1
                orig_list.append((orig_x, orig_y))
                orig_list_labels.append(
                    (orig_x - bregma_x, orig_y - bregma_y, orig_x, orig_y, num_label)
                )
                if (orig_x - bregma_x) < 0:
                    orig_list_labels_left.append(
                        (
                            orig_x - bregma_x,
                            orig_y - bregma_y,
                            orig_x,
                            orig_y,
                            num_label,
                        )
                    )
                elif (orig_x - bregma_x) > 0:
                    orig_list_labels_right.append(
                        (
                            orig_x - bregma_x,
                            orig_y - bregma_y,
                            orig_x,
                            orig_y,
                            num_label,
                        )
                    )
                orig_list.sort()
            orig_list_labels_sorted_left = sorted(
                orig_list_labels_left, key=lambda t: t[0], reverse=True
            )
            orig_list_labels_sorted_right = sorted(
                orig_list_labels_right, key=lambda t: t[0]
            )
            flatten = lambda l: [obj for sublist in l for obj in sublist]
            orig_list_labels_sorted = flatten(
                [orig_list_labels_sorted_left, orig_list_labels_sorted_right]
            )
            vertical_check = np.asarray([val[0] for val in orig_list_labels_sorted])
            for (orig_coord_val, orig_coord) in enumerate(orig_list_labels_sorted):
                vertical_close = np.where((abs(vertical_check - orig_coord[0]) <= 5))
                vertical_close_slice = vertical_close[0]
                vertical_matches = np.asarray(orig_list_labels_sorted)[
                    vertical_close_slice
                ]
                if len(vertical_close_slice) > 1:
                    vertical_match_sorted = sorted(vertical_matches, key=lambda t: t[1])
                    orig_list_labels_sorted_np = np.asarray(orig_list_labels_sorted)
                    orig_list_labels_sorted_np[
                        vertical_close_slice
                    ] = vertical_match_sorted
                    orig_list_labels_sorted = orig_list_labels_sorted_np.tolist()
            img = np.uint8(img)
        else:
            for num_label, cnt_orig in enumerate(cnts_orig):  # cnts_orig
                try:
                    cv2.drawContours(img, cnt_orig, -1, (255, 0, 0), 1)
                except:
                    print("Could not draw contour!")
        if not atlas_to_brain_align and use_unet:
            cortex_mask = cv2.imread(os.path.join(mask_path, "{}_mask.png".format(i)))
            cortex_mask = cv2.cvtColor(cortex_mask, cv2.COLOR_RGB2GRAY)
            thresh, cortex_mask_thresh = cv2.threshold(cortex_mask, 128, 255, 0)
            cortex_cnt = cv2.findContours(
                cortex_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cortex_cnt = imutils.grab_contours(cortex_cnt)
        labels_x = []
        labels_y = []
        areas = []
        sorted_labels_arr = []
        label_jitter = 0
        mask = np.zeros(mask_color.shape, dtype="uint8")
        cnts = cnts_orig
        print("LEN CNTS: {}".format(len(cnts)))
        print("LEN LABELS: {}".format(len(orig_list_labels_sorted)))
        if original_label or not atlas_to_brain_align:
            labels_from_region = [0] * len(orig_list_labels_sorted)
        for (z, cnt), (coord_idx, coord), label_from_region in zip(
            enumerate(cnts), enumerate(orig_list_labels_sorted), labels_from_region
        ):
            if atlas_to_brain_align and not original_label:
                coord_label_num = unique_regions.index(labels_from_region[coord_idx])
            else:
                coord_label_num = coord_idx
            # compute the center of the contour
            if len(cnts) > 1:
                z = 0
            c_x, c_y = int(coord[2]), int(coord[3])
            c = cnt
            if not atlas_to_brain_align and use_unet:
                cnt_loc_label = (
                    "inside"
                    if [1.0]
                    in [
                        list(
                            set(
                                [
                                    cv2.pointPolygonTest(
                                        cortex_sub_cnt,
                                        (
                                            c_coord.tolist()[0][0],
                                            c_coord.tolist()[0][1],
                                        ),
                                        False,
                                    )
                                    for c_coord in c
                                ]
                            )
                        )
                        for cortex_sub_cnt in cortex_cnt
                    ]
                    else "outside"
                )
            else:
                cnt_loc_label = ""
            rel_x = c_x - bregma_x
            rel_y = c_y - bregma_y

            pt_inside_cnt = [
                coord_check
                for coord_check in orig_list_labels_sorted
                if cv2.pointPolygonTest(
                    c, (int(coord_check[2]), int(coord_check[3])), False
                )
                == 1
            ]
            if original_label:
                try:
                    pt_inside_cnt_idx = orig_list_labels_sorted.index(pt_inside_cnt[0])
                    label_for_mat = pt_inside_cnt_idx
                except:
                    label_for_mat = coord_label_num
                    print(
                        "WARNING: label {} was not found in region. Order of labels may be incorrect!".format(str(coord_idx))
                    )
            else:
                label_for_mat = coord_label_num

            # if cnt_loc_label != '':
            #     coord_label_num = "{} {}".format(coord_label_num, cnt_loc_label)

            # The centroid of the contour works as the contour centre in most cases. However, sometimes the
            # centroid is outside of the contour. As such, using the average x and y positions of the contour edges
            # that intersect with the centroid could be a safer option. We try to find this average position and if
            # there are more than two intersecting edges or if the average position is over 200 px from the
            # centroid, we fall back to using the centroid as our measure of the centre of the contour.
            # for coord in c_for_centre:
            #     if coord[0][0] == c_x:
            #         edge_coords_y.append(coord[0].tolist())
            #     if coord[0][1] == c_y:
            #         edge_coords_x.append(coord[0].tolist())
            # print("{}: edge coords x: {}, edge coords y: {}".format(label, edge_coords_x, edge_coords_y))
            # adj_centre_x = int(np.mean([edge_coords_x[0][0], edge_coords_x[-1][0]]))
            # adj_centre_y = int(np.mean([edge_coords_y[0][1], edge_coords_y[-1][1]]))
            # adj_centre = [adj_centre_x, adj_centre_y]
            # if abs(adj_centre_x - c_x) <= 100 and abs(adj_centre_x - c_y) <= 100:
            #     print("adjusted centre: {}, {}".format(adj_centre[0], adj_centre[1]))
            #     c_x, c_y = (adj_centre[0], adj_centre[1])
            # edge_coords_x = []
            # edge_coords_y = []
            # compute center relative to bregma
            # rel_x = contour centre x coordinate - bregma x coordinate
            # rel_y = contour centre y coordinate - bregma y coordinate

            # print("Contour {}: centre ({}, {}), bregma ({}, {})".format(label, rel_x, rel_y, bregma_x, bregma_y))
            c_rel_centre = [rel_x, rel_y]
            if not os.path.isdir(
                os.path.join(segmented_save_path, "mat_contour_centre")
            ):
                os.mkdir(os.path.join(segmented_save_path, "mat_contour_centre"))

            # If .mat save checkbox checked in GUI, save contour paths and centre to .mat files for each contour
            if mat_save:
                mat_save = True
            else:
                mat_save = False
            # Prepares lists of the contours identified in the brain image, in the order that they are found by
            # OpenCV
            # labels_arr.append(label)
            sorted_labels_arr.append(coord_label_num)
            labels_x.append(int(c_x))
            labels_y.append(int(c_y))
            areas.append(cv2.contourArea(c))
            # The first contour just outlines the entire image (which does not provide a useful label or .mat
            # contour) so we'll ignore it
            # if coord_label_num != 0:
            # Cross-references each contour with a set of contours from a base brain atlas that was manually
            # labelled with brain regions (as defined in 'region_labels.csv' in the 'atlases' folder). If the
            # area of the contour is within 5000 square px of the original region and the centre of the contour
            # is at most 100 px away from the centre of the original contour, label the contour with its
            # corresponding brain region. Until we figure out how to consistently and accurately label small
            # brain regions, we only label brain regions with an area greater than 1000 square px.
            shape_list = []
            label_color = (0, 0, 255)
            for n_bc, bc in enumerate(base_c_max):
                shape_compare = cv2.matchShapes(c, bc, 1, 0.0)
                shape_list.append(shape_compare)
            # for (n_r, r), (n_bc, bc) in zip(enumerate(regions.itertuples()), enumerate(base_c_max)):
            #     min_bc = list(bc[0][0])
            #     min_c = list(c[0][0])
            #     max_bc = list(bc[0][-1])
            #     max_c = list(c[0][-1])
            #
            #     # 0.3, 75
            #     if label_num == 0 and region_labels and \
            #             (min(shape_list) - 0.3 <= cv2.matchShapes(c, bc, 1, 0.0) <= min(shape_list) + 0.3) and \
            #             min_bc[0] - 75 <= min_c[0] <= min_bc[0] + 75 and \
            #             min_bc[1] - 75 <= min_c[1] <= min_bc[1] + 75 and \
            #             max_bc[0] - 75 <= max_c[0] <= max_bc[0] + 75 and \
            #             max_bc[1] - 75 <= max_c[1] <= max_bc[1] + 75:
            #         # print("Current contour top left corner: {},{}".format(min_c[0], min_c[1]))
            #         # print("Baseline contour top left corner: {},{}".format(min_bc[0], min_bc[1]))
            #         closest_label = r.name
            #         cv2.putText(img, "{} ({})".format(closest_label, r.Index),
            #                     (int(c_x + label_jitter), int(c_y + label_jitter)),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color, 1)
            #         label_num += 1
            # if label_num == 0 and not region_labels:
            if (not region_labels and original_label) or (
                not region_labels and not atlas_to_brain_align
            ):
                (text_width, text_height) = cv2.getTextSize(
                    str(coord_label_num), cv2.FONT_HERSHEY_SIMPLEX, 0.4, thickness=1
                )[0]
                cv2.rectangle(
                    img,
                    (c_x + label_jitter, c_y + label_jitter),
                    (c_x + label_jitter + text_width, c_y + label_jitter - text_height),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    img,
                    str(coord_label_num),
                    (int(c_x + label_jitter), int(c_y + label_jitter)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    label_color,
                    1,
                )
                label_num += 1

            if mat_save:
                # Create an empty array of the same size as the contour, with the centre of the contour
                # marked as "255"
                c_total = np.zeros_like(mask)
                c_centre = np.zeros_like(mask)
                # Follow the path of the contour, setting every pixel along the path to 255
                # Fill in the contour area with 1s
                cv2.fillPoly(c_total, pts=[c], color=(255, 255, 255))
                # Set the contour's centroid to 255
                if c_x < mask.shape[0] and c_y < mask.shape[0]:
                    c_centre[c_x, c_y] = 255
                if not os.path.isdir(os.path.join(segmented_save_path, "mat_contour")):
                    os.mkdir(os.path.join(segmented_save_path, "mat_contour"))
                if not os.path.isdir(
                    os.path.join(segmented_save_path, "mat_contour_centre")
                ):
                    os.mkdir(os.path.join(segmented_save_path, "mat_contour_centre"))
                sio.savemat(
                    os.path.join(
                        segmented_save_path,
                        "mat_contour/roi_{}_{}_{}_{}.mat".format(
                            cnt_loc_label, i, label_for_mat, z
                        ),
                    ),
                    {
                        "roi_{}_{}_{}_{}".format(
                            cnt_loc_label, i, label_for_mat, z
                        ): c_total
                    },
                    appendmat=False,
                )
                sio.savemat(
                    os.path.join(
                        segmented_save_path,
                        "mat_contour_centre/roi_centre_{}_{}_{}_{}.mat".format(
                            cnt_loc_label, i, label_for_mat, z
                        ),
                    ),
                    {
                        "roi_centre_{}_{}_{}_{}".format(
                            cnt_loc_label, i, label_for_mat, z
                        ): c_centre
                    },
                    appendmat=False,
                )
                sio.savemat(
                    os.path.join(
                        segmented_save_path,
                        "mat_contour_centre/rel_roi_centre_{}_{}_{}_{}.mat".format(
                            cnt_loc_label, i, label_for_mat, z
                        ),
                    ),
                    {
                        "rel_roi_centre_{}_{}_{}_{}".format(
                            cnt_loc_label, i, label_for_mat, z
                        ): c_rel_centre
                    },
                    appendmat=False,
                )
            count += 1
        if align_once:
            idx_to_use = 0
        else:
            idx_to_use = i
        if plot_landmarks:
            for pt, color in zip(dlc_pts[idx_to_use], colors):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 10, color, -1)
            for pt, color in zip(atlas_pts[idx_to_use], colors):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 5, color, -1)
        io.imsave(
            os.path.join(segmented_save_path, "{}_mask_segmented.png".format(i)), img
        )
        img_edited = Image.open(os.path.join(save_path, "{}_mask_binary.png".format(i)))
        # Generates a transparent version of the brain atlas.
        img_rgba = img_edited.convert("RGBA")
        data = img_rgba.getdata()
        for pixel in data:
            if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
                new_data.append((pixel[0], pixel[1], pixel[2], 0))
            else:
                new_data.append(pixel)
        img_rgba.putdata(new_data)
        img_rgba.save(os.path.join(save_path, "{}_mask_transparent.png".format(i)))
        img_transparent = cv2.imread(
            os.path.join(save_path, "{}_mask_transparent.png".format(i))
        )
        img_trans_for_mat = np.uint8(img_transparent)
        if mat_save:
            sio.savemat(
                os.path.join(
                    segmented_save_path, "mat_contour/transparent_{}".format(i)
                ),
                {"transparent_{}".format(i): img_trans_for_mat},
            )
        masked_img = cv2.bitwise_and(img, img_transparent, mask=mask_color)
        if plot_landmarks:
            for pt, color in zip(dlc_pts[idx_to_use], colors):
                cv2.circle(masked_img, (int(pt[0]), int(pt[1])), 10, color, -1)
            for pt, color in zip(atlas_pts[idx_to_use], colors):
                cv2.circle(masked_img, (int(pt[0]), int(pt[1])), 5, color, -1)
        io.imsave(os.path.join(save_path, "{}_overlay.png".format(i)), masked_img)
        print("Mask {} saved!".format(i))
        d = {
            "sorted_label": sorted_labels_arr,
            "x": labels_x,
            "y": labels_y,
            "area": areas,
        }
        df = pd.DataFrame(data=d)
        if not os.path.isdir(os.path.join(segmented_save_path, "region_labels")):
            os.mkdir(os.path.join(segmented_save_path, "region_labels"))
        df.to_csv(
            os.path.join(
                segmented_save_path, "region_labels", "{}_region_labels.csv".format(i)
            )
        )
    print(
        "Analysis complete! Check the outputs in the folders of {}.".format(save_path)
    )
    k.clear_session()
    if dlc_pts:
        os.chdir("../..")
    else:
        os.chdir(os.path.join(save_path, '..'))
