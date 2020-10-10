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


def testGenerator(test_path, output_mask_path, num_image=60, target_size=(512, 512), flag_multi_class=False,
                  as_gray=True, atlas_to_brain_align=True):
    """
    Import images and resize it to the target size of the model.
    :param test_path: path to input images
    :param num_image: number of input images
    :param target_size: target image size as defined in the Keras model
    :param flag_multi_class: flag the input images as having multiple classes
    :param as_gray: if input image is grayscale, process data input accordingly
    """
    suff = 'png'
    print(test_path)
    img_list = glob.glob(os.path.join(test_path, "*png"))
    img_list.sort(key=natural_sort_key)
    tif_list = glob.glob(os.path.join(test_path, "*tif"))
    print(tif_list)
    if tif_list:
        print(tif_list)
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
                img = io.imread(os.path.join(test_path, "{}_brain_warp.{}".format(i, suff)))
            img = trans.resize(img, target_size)
        img = img_as_ubyte(img)
        io.imsave(os.path.join(output_mask_path, "{}.{}".format(i, suff)), img)
        img = io.imread(os.path.join(output_mask_path, "{}.{}".format(i, suff)), as_gray=as_gray)
        if img.dtype == 'uint8':
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
    :param num_class: if flag_multi_class is True, decide how many classes to categorize each brain image into.
    """
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        io.imsave(os.path.join(save_path, "{}.png".format(i)), img)


def atlas_to_mask(atlas_path, mask_input_path, mask_warped_path, mask_output_path, n, use_unet,
                  atlas_to_brain_align, git_repo_base, olfactory_check):
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
    """
    atlas = cv2.imread(atlas_path, cv2.IMREAD_GRAYSCALE)
    mask_warped = cv2.imread(mask_warped_path, cv2.IMREAD_GRAYSCALE)
    # print(mask_warped_path)
    # print(use_unet)
    if use_unet:
        mask_input = cv2.imread(mask_input_path, cv2.IMREAD_GRAYSCALE)
        if olfactory_check:
            cnts_for_olfactory = cv2.findContours(mask_input.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
            cnts_for_olfactory = imutils.grab_contours(cnts_for_olfactory)
        io.imsave(os.path.join(mask_output_path, "{}_mask.png".format(n)), mask_input)
        # Adds the common white regions of the atlas and U-net mask together into a binary image.
        if atlas_to_brain_align:
            # FOR ALIGNING ATLAS TO BRAIN
            mask_input = cv2.bitwise_and(atlas, mask_input)
            mask_input = cv2.bitwise_and(mask_input, mask_warped)
            if olfactory_check:
                olfactory_bulbs = sorted(cnts_for_olfactory, key=cv2.contourArea, reverse=True)[2:4]
                for bulb in olfactory_bulbs:
                    cv2.fillPoly(mask_input, pts=[bulb], color=[255, 255, 255])
        else:
            # FOR ALIGNING BRAIN TO ATLAS
            mask_input = cv2.bitwise_and(atlas, mask_warped)
    else:
        mask_input = cv2.bitwise_and(atlas, mask_warped)
        if olfactory_check:
            olfactory_path = os.path.join(git_repo_base, 'atlases')
            olfactory_left = cv2.imread(os.path.join(olfactory_path, '02.png'), cv2.IMREAD_GRAYSCALE)
            olfactory_right = cv2.imread(os.path.join(olfactory_path, '01.png'), cv2.IMREAD_GRAYSCALE)
            olfactory_mask_left, cnts_for_olfactory_left, hierarchy = cv2.findContours(olfactory_left, cv2.RETR_TREE,
                                                                            cv2.CHAIN_APPROX_NONE)
            # print("LEN OLFACTORY LEFT: {}".format(len(cnts_for_olfactory_left)))
            olfactory_left_cnt = min(cnts_for_olfactory_left, key=cv2.contourArea)
            olfactory_mask_right, cnts_for_olfactory_right, hierarchy = cv2.findContours(olfactory_right, cv2.RETR_TREE,
                                                                            cv2.CHAIN_APPROX_NONE)
            olfactory_right_cnt = min(cnts_for_olfactory_right, key=cv2.contourArea)
            cv2.fillPoly(mask_input, pts=[olfactory_left_cnt], color=[255, 255, 255])
            cv2.fillPoly(mask_input, pts=[olfactory_right_cnt], color=[255, 255, 255])
    io.imsave(os.path.join(mask_output_path, "{}.png".format(n)), mask_input)


def inpaintMask(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    for cnt in cnts:
        cv2.fillPoly(mask, pts=[cnt], color=[255, 255, 255])
    return mask


def applyMask(image_path, mask_path, save_path, segmented_save_path, mat_save, threshold, git_repo_base, bregma_list,
              atlas_to_brain_align, model, dlc_pts, atlas_pts, olfactory_check, use_unet, plot_landmarks, align_once,
              region_labels=True):
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
    :param align_once: if True, carries out all alignments based on the alignment of the first atlas and brain. This can
    save time if you have many frames of the same brain with a fixed camera position.
    :param region_labels: choose whether to est
    """
    # print(bregma_list)
    tif_list = glob.glob(os.path.join(image_path, "*tif"))
    if atlas_to_brain_align:
        image_name_arr = glob.glob(os.path.join(image_path, "*.png"))
        image_name_arr.sort(key=natural_sort_key)
        if tif_list:
            # print(tif_list)
            tif_stack = imageio.mimread(os.path.join(image_path, tif_list[0]))
            num_images = len(tif_stack)
            # print(num_images)
            image_name_arr = tif_stack
    else:
        # FOR ALIGNING BRAIN TO ATLAS
        # image_name_arr = glob.glob(os.path.join(mask_path, "*_atlas.png"))
        image_name_arr = glob.glob(os.path.join(mask_path, "*_brain_warp.png"))
        image_name_arr.sort(key=natural_sort_key)


    region_bgr_lower = (220, 220, 220)
    region_bgr_upper = (255, 255, 255)
    base_c_max = []
    count = 0
    # regions = pd.read_csv(os.path.join(git_repo_base, "atlases/region_labels.csv"))
    # Find the contours of an existing set of brain regions (to be used to identify each new brain region by shape)
    mat_files = glob.glob(os.path.join(git_repo_base, 'atlases/mat_contour_base/*.mat'))
    mat_files.sort(key=natural_sort_key)
    # colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0)]

    # adapted from https://stackoverflow.com/questions/3016283/create-a-color-generator-from-given-colormap-in-matplotlib
    cm = pylab.get_cmap('viridis')
    colors = [cm(1. * i / NUM_COLORS)[0:3] for i in range(NUM_COLORS)]
    colors = [tuple(color_idx * 255 for color_idx in color_t) for color_t in colors]
    # vertical_aligns = []
    for file in mat_files:
        mat = scipy.io.loadmat(os.path.join(git_repo_base, 'atlases/mat_contour_base/', file))
        mat = mat['vect']
        ret, thresh = cv2.threshold(mat, 5, 255, cv2.THRESH_BINARY)
        base_c = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        base_c = imutils.grab_contours(base_c)
        base_c_max.append(max(base_c, key=cv2.contourArea))
    if not atlas_to_brain_align and use_unet:
        # FOR ALIGNING ATLAS TO BRAIN
        num_images = len(glob.glob(os.path.join(mask_path, '*_brain_warp*')))
        output = os.path.join(mask_path, '..')
        # print(output)
        from mesonet.predict_regions import predictRegion
        mask_generate = True
        tif_list = glob.glob(os.path.join(image_path, "*tif"))
        # print(tif_list)
        if tif_list:
            input_path = image_path
        else:
            input_path = mask_path
        predictRegion(input_path, num_images, model, output, mat_save, threshold, mask_generate, git_repo_base,
                      atlas_to_brain_align, dlc_pts, atlas_pts, olfactory_check, use_unet, plot_landmarks, align_once,
                      region_labels)
    for i, item in enumerate(image_name_arr):
        label_num = 0
        if not atlas_to_brain_align:
            atlas_path = os.path.join(mask_path, '{}_atlas.png'.format(str(i)))
            mask_input_path = os.path.join(mask_path, '{}.png'.format(i))
            mask_warped_path = os.path.join(mask_path, '{}_mask_warped.png'.format(str(i)))
            atlas_to_mask(atlas_path, mask_input_path, mask_warped_path, mask_path, i, use_unet,
                          atlas_to_brain_align, git_repo_base, olfactory_check)
        bregma_x, bregma_y = bregma_list[i]
        new_data = []
        if len(tif_list) != 0 and atlas_to_brain_align:
            img = item
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # print(item)
            img = cv2.imread(item)
        if atlas_to_brain_align:
            img = cv2.resize(img, (512, 512))
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
        atlas_bw = cv2.dilate(atlas_bw, kernel, iterations=1)  # 1
        io.imsave(os.path.join(save_path, "{}_atlas_binary.png".format(i)), atlas_bw)

        if not atlas_to_brain_align:
            watershed_run_rule = i == 0
            # watershed_run_rule = True
        else:
            if len(tif_list) == 0:
                watershed_run_rule = True
            else:
                watershed_run_rule = i == 0
        if align_once:
            watershed_run_rule = i == 0

        if watershed_run_rule:
            orig_list = []
            orig_list_labels = []
            orig_list_labels_left = []
            orig_list_labels_right = []
            # Find contours in original aligned atlas
            cnts_orig = cv2.findContours(atlas_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts_orig = imutils.grab_contours(cnts_orig)
            # print("CNTS ORIG: {}".format(len(cnts_orig)))
            labels_cnts = []
            for num_label, cnt_orig in enumerate(cnts_orig):  # cnts_orig
                # cnt_orig_moment = cv2.moments(cnt_orig)
                labels_cnts.append(cnt_orig)
                try:
                    cv2.drawContours(img, cnt_orig, -1, (255, 0, 0), 1)
                except:
                    print("Could not draw contour!")
                # print(num_label)
                if num_label not in [0, 1]:
                    try:
                        c_orig_as_list = cnt_orig.tolist()
                        c_orig_as_list = [[c_val[0] for c_val in c_orig_as_list]]
                        orig_polylabel = polylabel(c_orig_as_list)
                        # print(orig_polylabel)
                        orig_x, orig_y = int(orig_polylabel[0]), int(orig_polylabel[1])
                        orig_list.append((orig_x, orig_y))
                        orig_list_labels.append((orig_x - bregma_x, orig_y - bregma_y, orig_x, orig_y, num_label))
                        if (orig_x - bregma_x) < 0:
                            orig_list_labels_left.append(
                                (orig_x - bregma_x, orig_y - bregma_y, orig_x, orig_y, num_label))
                        elif (orig_x - bregma_x) > 0:
                            orig_list_labels_right.append(
                                (orig_x - bregma_x, orig_y - bregma_y, orig_x, orig_y, num_label))
                        # orig_list_labels.append((orig_x - (np.shape(atlas_bw)[0])/2, orig_y - (np.shape(atlas_bw)[1])/2,
                        #                          orig_x, orig_y, num_label))
                        # orig_x = int(cnt_orig_moment["m10"] / cnt_orig_moment["m00"])
                        # orig_y = int(cnt_orig_moment["m01"] / cnt_orig_moment["m00"])
                        # for coord in cnt_orig:
                        #     if coord[0][0] == orig_x:
                        #         edge_coords_orig_y.append(coord[0].tolist())
                        #     if coord[0][1] == orig_y:
                        #         edge_coords_orig_x.append(coord[0].tolist())
                        # # print("{}: edge coords x: {}, edge coords y: {}".format(num_label, edge_coords_orig_x,
                        # edge_coords_orig_y))
                        # adj_centre_x = int(np.mean([edge_coords_orig_x[0][0], edge_coords_orig_x[-1][0]]))
                        # adj_centre_y = int(np.mean([edge_coords_orig_y[0][1], edge_coords_orig_y[-1][1]]))
                        # adj_centre = [adj_centre_x, adj_centre_y]
                        # if abs(adj_centre_x - orig_x) <= 100 and abs(adj_centre_x - orig_y) <= 100:
                        #     # print("adjusted centre: {}, {}".format(adj_centre[0], adj_centre[1]))
                        #     orig_x, orig_y = (adj_centre[0], adj_centre[1])
                        # edge_coords_orig_x = []
                        # edge_coords_orig_y = []
                        # cv2.putText(img, str(num_label),
                        #             (int(orig_x), int(orig_y)),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    except:
                        print("cannot find moments!")
                orig_list.sort()
            # orig_list_labels_sorted = sorted(orig_list_labels, key=lambda t: t[0])
            orig_list_labels_sorted_left = sorted(orig_list_labels_left, key=lambda t: t[0], reverse=True)
            orig_list_labels_sorted_right = sorted(orig_list_labels_right, key=lambda t: t[0])
            flatten = lambda l: [obj for sublist in l for obj in sublist]
            orig_list_labels_sorted = flatten([orig_list_labels_sorted_left, orig_list_labels_sorted_right])
            # print(orig_list_labels_sorted)
            vertical_check = np.asarray([val[0] for val in orig_list_labels_sorted])
            # if len(vertical_aligns) > 0:
            #     # print(vertical_aligns)
            #     orig_list_labels_sorted_np = np.asarray(orig_list_labels_sorted)
            #     # print(orig_list_labels_sorted_np[vertical_close_slice])
            #     for v_align_slice in vertical_aligns:
            #         prev_vertical_matches = np.asarray(orig_list_labels_sorted)[v_align_slice]
            #         orig_list_labels_sorted_np[v_align_slice] = prev_vertical_matches
            #         orig_list_labels_sorted = orig_list_labels_sorted_np.tolist()
            for (orig_coord_val, orig_coord) in enumerate(orig_list_labels_sorted):
                vertical_close = np.where((abs(vertical_check - orig_coord[0]) <= 5))
                # print(vertical_close)
                vertical_close_slice = vertical_close[0]
                vertical_matches = np.asarray(orig_list_labels_sorted)[vertical_close_slice]
                # print(vertical_matches)
                if len(vertical_close_slice) > 1:
                    # vertical_aligns.append(vertical_close_slice)
                    # print(vertical_aligns)
                    vertical_match_sorted = sorted(vertical_matches, key=lambda t: t[1])
                    orig_list_labels_sorted_np = np.asarray(orig_list_labels_sorted)
                    # print(orig_list_labels_sorted_np[vertical_close_slice])
                    orig_list_labels_sorted_np[vertical_close_slice] = vertical_match_sorted
                    # print(orig_list_labels_sorted_np[vertical_close_slice])
                    orig_list_labels_sorted = orig_list_labels_sorted_np.tolist()
                    # print(orig_list_labels_sorted)
            # print(orig_list_labels_sorted[0:20])
            # opening = cv2.morphologyEx(atlas_bw, cv2.MORPH_OPEN, kernel, iterations=1)  # 1
            # opening = mask_color
            # opening = cv2.dilate(mask_color, kernel, iterations=2)
            # opening = cv2.erode(mask_color, kernel, iterations=1)
            # io.imsave(os.path.join(mask_path, 'opening_test_{}.png'.format(i)), opening)
            # sure background area
            # kernel = np.ones((3, 3), np.uint8)  # 3, 3
            # sure_bg = cv2.erode(atlas_bw, kernel, iterations=1)  # 7, 5
            # Finding sure foreground area
            # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # 5
            # dist_transform = np.uint8(dist_transform)
            # ret, sure_fg = cv2.threshold(dist_transform, threshold * dist_transform.max(), 255, 0)
            # Finding unknown region
            # sure_fg = atlas_bw
            # sure_fg = np.uint8(sure_fg)
            # sure_fg = cv2.erode(sure_fg, kernel, iterations=2)
            # unknown = cv2.subtract(sure_fg, sure_bg)
            # io.imsave(os.path.join(mask_path, 'opening_test_{}.png'.format(i)), unknown)
            # ret, markers = cv2.connectedComponents(sure_fg)
            # Add one to all labels so that sure background is not 0, but 1
            # markers = markers + 1
            # Now, mark the region of unknown with zero
            # markers[unknown == 255] = 0
            # io.imsave(os.path.join(mask_path, 'markers_{}.png'.format(i)), unknown)
            # io.imsave(os.path.join(mask_path, 'markers_{}.png'.format(i)), markers)
            img = np.uint8(img)
            # labels = cv2.watershed(img, markers)
            # io.imsave(os.path.join(mask_path, 'labels_{}.png'.format(i)), labels)
            # print(labels)
        else:
            for num_label, cnt_orig in enumerate(cnts_orig):  # cnts_orig
                try:
                    cv2.drawContours(img, cnt_orig, -1, (255, 0, 0), 1)
                except:
                    print("Could not draw contour!")
        if not atlas_to_brain_align and use_unet:
            cortex_mask = cv2.imread(os.path.join(mask_path, "{}_mask.png".format(i)))
            cortex_mask = cv2.cvtColor(cortex_mask, cv2.COLOR_RGB2GRAY)
            # cortex_mask = np.uint8(cortex_mask)
            thresh, cortex_mask_thresh = cv2.threshold(cortex_mask, 128, 255, 0)
            cortex_cnt = cv2.findContours(cortex_mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cortex_cnt = imutils.grab_contours(cortex_cnt)
            cv2.drawContours(img, cortex_cnt, -1, (0, 0, 255), 3)
        # img[labels == -1] = [255, 0, 0]
        sorted_nums = np.argsort(orig_list_labels_sorted)
        # print("sorted nums: {}".format(sorted_nums))
        # print("labels: {}".format(labels))
        # labels = [x for _, x in sorted(zip(sorted_nums[0], labels))]
        labels_x = []
        labels_y = []
        areas = []
        # labels_arr = []
        sorted_labels_arr = []
        # labels_cnts = []
        # label_jitter = random.randrange(-2, 2)
        label_jitter = 0
        # count_label = 0
        # print("LABELS ORIG: {}".format(len(np.unique(labels))))
        mask = np.zeros(mask_color.shape, dtype="uint8")
        # for (n, label) in enumerate(np.unique(labels)):
        #     # label_num = 0
        #     # if the label is zero, we are examining the 'background'
        #     # so simply ignore it
        #     # if label <= 0:
        #     #   continue
        #     # otherwise, allocate memory for the label region and draw
        #     # it on the mask
        #     mask = np.zeros(mask_color.shape, dtype="uint8")
        #     if atlas_to_brain_align:
        #         mask[labels == label] = 255
        #     # mask_dilate = np.zeros(mask_color.shape, dtype="uint8")
        #     # detect contours in the mask and grab the largest one
        #     if atlas_to_brain_align:
        #         cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        #                                 cv2.CHAIN_APPROX_NONE)
        #         cnts = imutils.grab_contours(cnts)
        #         if len(cnts) > 0:
        #             c = max(cnts, key=cv2.contourArea)
        #             labels_cnts.append(c)
        #     # print("OUTER LOOP")
        # if not atlas_to_brain_align:
        #     cnts = mat_cnt_list
        # else:
        #     cnts = cnts_orig
        cnts = cnts_orig
        # print("LEN CNTS: {}".format(len(cnts)))
        for (z, cnt), (coord_label_num, coord) in zip(enumerate(cnts),
                                                      enumerate(orig_list_labels_sorted)):
            # label = str(coord_label_num)
            # compute the center of the contour
            if len(cnts) > 1:
                z = 0
            c_x, c_y = int(coord[2]), int(coord[3])
            # c_for_centre = min(inner_cnts, key=cv2.contourArea)
            # m = cv2.moments(cnt)
            # m = cv2.moments(c_for_centre)
            # c_x = int(m["m10"] / m["m00"])
            # c_y = int(m["m01"] / m["m00"])
            c = cnt
            # c = max(cnts, key=cv2.contourArea)
            if not atlas_to_brain_align and use_unet:
                # if i != 0:
                #    cv2.drawContours(img, c, -1, (255, 0, 0), 1)
                # print([(c_coord.tolist()[0][0], c_coord.tolist()[0][1]) for c_coord in c])
                # print([list(set([cv2.pointPolygonTest(cortex_sub_cnt, (c_coord.tolist()[0][0],
                #                                                        c_coord.tolist()[0][1]), False)
                #                 for c_coord in c])) for cortex_sub_cnt in cortex_cnt])
                # print([cv2.pointPolygonTest(cortex_cnt, (c_coord[0][0], c_coord[0][1]), False) for c_coord in c])
                cnt_loc_label = "inside" if [1.0] in [list(set([cv2.pointPolygonTest(cortex_sub_cnt,
                                                                                     (c_coord.tolist()[0][0],
                                                                                      c_coord.tolist()[0][1]),
                                                                                     False)
                                                                for c_coord in c])) for cortex_sub_cnt in
                                                      cortex_cnt] else "outside"
            else:
                cnt_loc_label = ""
            # c_as_list = c.tolist()
            # c_as_list = [[c_val[0] for c_val in c_as_list]]
            # centre_polylabel = polylabel(c_as_list)
            # c_x, c_y = int(centre_polylabel[0]), int(centre_polylabel[1])
            rel_x = c_x - bregma_x
            rel_y = c_y - bregma_y
            # rel_x = coord[0]
            # rel_y = coord[1]
            # print([item for item in orig_list_labels_sorted if rel_x in item])
            # label = [orig_list_labels_sorted.index(item) for item in orig_list_labels_sorted if rel_x in item][0]
            # print('label: {}'.format(label))
            # print('coord_label_num: {}'.format(coord_label_num))
            pt_inside_cnt = [coord_check for coord_check in orig_list_labels_sorted if
                             cv2.pointPolygonTest(c, (int(coord_check[2]), int(coord_check[3])), False) == 1]
            # print(pt_inside_cnt)
            try:
                pt_inside_cnt_idx = orig_list_labels_sorted.index(pt_inside_cnt[0])
                # print(pt_inside_cnt_idx)
                label_for_mat = pt_inside_cnt_idx
            except:
                label_for_mat = coord_label_num
                print("WARNING: label was not found in region. Order of labels may be incorrect!")

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
            if not os.path.isdir(os.path.join(segmented_save_path, 'mat_contour_centre')):
                os.mkdir(os.path.join(segmented_save_path, 'mat_contour_centre'))

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
            if not region_labels:
                (text_width, text_height) = cv2.getTextSize(str(coord_label_num), cv2.FONT_HERSHEY_SIMPLEX,
                                                            0.4, thickness=1)[0]
                cv2.rectangle(img, (c_x + label_jitter, c_y + label_jitter),
                              (c_x + label_jitter + text_width, c_y + label_jitter - text_height),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(img, str(coord_label_num),
                            (int(c_x + label_jitter), int(c_y + label_jitter)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
                label_num += 1
                # print(label_num)
                # print(count_label)
                # n > 0
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
                    if not os.path.isdir(os.path.join(segmented_save_path, 'mat_contour')):
                        os.mkdir(os.path.join(segmented_save_path, 'mat_contour'))
                    if not os.path.isdir(os.path.join(segmented_save_path, 'mat_contour_centre')):
                        os.mkdir(os.path.join(segmented_save_path, 'mat_contour_centre'))
                    sio.savemat(os.path.join(segmented_save_path,
                                             'mat_contour/roi_{}_{}_{}_{}.mat'.format(cnt_loc_label,
                                                                                      i, label_for_mat, z)),
                                {'roi_{}_{}_{}_{}'.format(cnt_loc_label,
                                                          i, label_for_mat, z): c_total}, appendmat=False)
                    sio.savemat(os.path.join(segmented_save_path,
                                             'mat_contour_centre/roi_centre_{}_{}_{}_{}.mat'.format(
                                                 cnt_loc_label, i, label_for_mat, z)),
                                {'roi_centre_{}_{}_{}_{}'.format(cnt_loc_label, i, label_for_mat,
                                                                 z): c_centre},
                                appendmat=False)
                    sio.savemat(os.path.join(segmented_save_path,
                                             'mat_contour_centre/rel_roi_centre_{}_{}_{}_{}.mat'.format(
                                                 cnt_loc_label, i, label_for_mat, z)),
                                {'rel_roi_centre_{}_{}_{}_{}'.format(cnt_loc_label, i,
                                                                     label_for_mat, z): c_rel_centre},
                                appendmat=False)
            count += 1
        # print(plot_landmarks)
        if plot_landmarks:
            for pt, color in zip(dlc_pts[i], colors):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 10, color, -1)
            for pt, color in zip(atlas_pts[i], colors):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 5, color, -1)
        io.imsave(os.path.join(segmented_save_path, "{}_mask_segmented.png".format(i)), img)
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
        img_transparent = cv2.imread(os.path.join(save_path, "{}_mask_transparent.png".format(i)))
        img_trans_for_mat = np.uint8(img_transparent)
        if mat_save:
            sio.savemat(os.path.join(segmented_save_path, 'mat_contour/transparent_{}'.format(i)),
                        {'transparent_{}'.format(i): img_trans_for_mat})
        masked_img = cv2.bitwise_and(img, img_transparent, mask=mask_color)
        if plot_landmarks:
            print(dlc_pts[i])
            for pt, color in zip(dlc_pts[i], colors):
                cv2.circle(masked_img, (int(pt[0]), int(pt[1])), 10, color, -1)
            print(atlas_pts[i])
            for pt, color in zip(atlas_pts[i], colors):
                cv2.circle(masked_img, (int(pt[0]), int(pt[1])), 5, color, -1)
                print((pt[0] - atlas_pts[i][4][0], pt[1] - atlas_pts[i][4][1]))
        io.imsave(os.path.join(save_path, "{}_overlay.png".format(i)), masked_img)
        print("Mask {} saved!".format(i))
        d = {'sorted_label': sorted_labels_arr, 'x': labels_x, 'y': labels_y, 'area': areas}
        df = pd.DataFrame(data=d)
        if not os.path.isdir(os.path.join(segmented_save_path, 'region_labels')):
            os.mkdir(os.path.join(segmented_save_path, 'region_labels'))
        df.to_csv(os.path.join(segmented_save_path, 'region_labels', '{}_region_labels.csv'.format(i)))
    print('Analysis complete! Check the outputs in the folders of {}.'.format(save_path))
    k.clear_session()
    os.chdir('../..')
