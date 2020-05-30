"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
from mesonet.mask_functions import atlas_to_mask, applyMask
import numpy as np
import pandas as pd
import cv2
import scipy.io
import skimage.io as io
import skimage.transform as trans
from skimage.transform import PiecewiseAffineTransform, warp
import os
import sys
import fnmatch
import glob


def find_peaks(img):
    """
    Locates the peaks of cortical activity in each hemisphere of the brain image.
    :param img: A sensory map (indicating functional activation in the mouse brain).
    :return: An array of the maximum points of activation for the sensory map.
    """
    maxLocArr = []
    img = cv2.imread(str(img), 0)
    im = img.copy()
    x_min = int(np.around(im.shape[0]/2))
    im1 = im[:, x_min:im.shape[0]]
    im2 = im[:, 0:x_min]
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(im1)
    (minVal2, maxVal2, minLoc2, maxLoc2) = cv2.minMaxLoc(im2)
    maxLoc = list(maxLoc)
    maxLoc[0] = maxLoc[0] + x_min
    maxLoc = tuple(maxLoc)
    maxLocArr.append(maxLoc)
    if (maxVal - 30) <= maxVal2 <= (maxVal + 30):
        maxLocArr.append(maxLoc2)
    return maxLocArr


def coords_to_mat(sub_pts, i, output_mask_path, bregma_present, bregma_index, landmark_arr):
    if bregma_present:
        x_bregma, y_bregma = sub_pts[bregma_index]
        for pt, landmark in zip(sub_pts, landmark_arr):
            x_pt = pt[0]
            y_pt = pt[1]
            pt_adj = [landmark, x_pt - x_bregma, y_pt - y_bregma]
            pt_adj_to_mat = np.array(pt_adj, dtype=np.object)
            print("landmark position:{}".format(pt_adj))
            if not os.path.isdir(os.path.join(output_mask_path, 'mat_coords')):
                os.mkdir(os.path.join(output_mask_path, 'mat_coords'))
            scipy.io.savemat(os.path.join(output_mask_path,
                                     'mat_coords/landmarks_{}_{}.mat'.format(i, landmark)),
                        {'landmark_coords_{}_{}'.format(i, landmark): pt_adj_to_mat}, appendmat=False)


def sensory_to_mat(sub_pts, bregma_pt, i, output_mask_path):
    x_bregma, y_bregma = bregma_pt
    sensory_names = ['tail_left', 'tail_right', 'visual', 'whisker']
    for pt, landmark in zip(sub_pts, sensory_names):
        x_pt = pt[0]
        y_pt = pt[1]
        pt_adj = [landmark, x_pt - x_bregma, y_pt - y_bregma]
        pt_adj_to_mat = np.array(pt_adj, dtype=np.object)
        if not os.path.isdir(os.path.join(output_mask_path, 'mat_coords')):
            os.mkdir(os.path.join(output_mask_path, 'mat_coords'))
        scipy.io.savemat(os.path.join(output_mask_path,
                                 'mat_coords/sensory_peaks_{}_{}.mat'.format(i, landmark)),
                    {'sensory_peaks_{}_{}'.format(i, landmark): pt_adj_to_mat}, appendmat=False)


def atlas_from_mat(input_file):
    """
    Generates a binary brain atlas from a .mat file.
    :param input_file: The input .mat file representing a brain atlas (with white = 255 and black = 0)
    :return: A thresholded binary brain atlas.
    """
    file = input_file
    mat = scipy.io.loadmat(file)
    mat = mat['cdata']
    ret, thresh = cv2.threshold(mat, 5, 255, cv2.THRESH_BINARY_INV)
    return thresh


def getMaskContour(mask_dir, atlas_img, predicted_pts, actual_pts, cwd, n, main_mask):
    """
    Gets the contour of the brain's boundaries and applies a piecewise affine transform to the brain atlas
    based on the cortical landmarks predicted in dlc_predict (and peaks of activity on the sensory map, if available).
    :param mask_dir: The path to the directory containing the U-net masks of the brain's boundaries.
    :param atlas_img: The brain atlas to be transformed.
    :param predicted_pts: The coordinates of the cortical landmarks predicted in dlc_predict (or, for the second run
    of this function, the coordinates of the peaks of activity in the sensory map).
    :param actual_pts: The fixed coordinates of the cortical landmarks on the brain atlas (or, for the second run of
    this function, the fixed coordinates of the peaks of sensory activity on the brain atlas).
    :param cwd: The path to the current working directory.
    :param n: The number of the current image in the directory.
    """
    c_landmarks = np.empty([0, 2])
    c_atlas_landmarks = np.empty([0, 2])
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    atlas_to_warp = atlas_img
    mask = np.uint8(mask)
    mask_new, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        cnt = cnt[:, 0, :]
        cnt = np.asarray(cnt).astype('float32')
        c_landmarks = np.concatenate((c_landmarks, cnt))
        c_atlas_landmarks = np.concatenate((c_atlas_landmarks, cnt))
    c_landmarks = np.concatenate((c_landmarks, predicted_pts))
    c_atlas_landmarks = np.concatenate((c_atlas_landmarks, actual_pts))
    tform = PiecewiseAffineTransform()
    tform.estimate(c_atlas_landmarks, c_landmarks)
    dst = warp(atlas_to_warp, tform, output_shape=(512, 512))
    if main_mask:
        io.imsave(os.path.join(cwd, "mask_{}.png".format(n)), mask)
    return dst


def atlasBrainMatch(brain_img_dir, sensory_img_dir, coords_input, sensory_match,
                    mat_save, threshold, git_repo_base, region_labels, landmark_arr):
    """
    Align and overlap brain atlas onto brain image based on four landmark locations in the brain image and the atlas.
    :param brain_img_dir: The directory containing each brain image to be used.
    :param sensory_img_dir: The directory containing each sensory image to be used (if you are aligning each brain
    image using a sensory map).
    :param coords_input: Predicted locations of the four landmarks on the brain image from the file generated by
    DeepLabCut.
    :param sensory_match: Whether or not a sensory map is to be used.
    :param mat_save: Whether or not to export each brain region to a .mat file in applyMask, which is called at the end
    of this function.
    :param threshold: The threshold for the cv2.opening operation carried out in applyMask, which is called at the end
    of this function.
    :param git_repo_base: The path to the base git repository containing necessary resources for MesoNet (reference
    atlases, DeepLabCut config files, etc.)
    :param region_labels: Choose whether or not to attempt to label each region with its name from the Allen Institute
    Mouse Brain Atlas.
    """
    # load brain images folder
    brain_img_arr = []
    dlc_img_arr = []
    peak_arr = []

    # Prepare output folder
    cwd = os.getcwd()
    output_mask_path = os.path.join(cwd, "../output_mask")
    # Output folder for transparent masks and masks overlaid onto brain image
    output_overlay_path = os.path.join(cwd, "../output_overlay")
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    if not os.path.isdir(output_overlay_path):
        os.mkdir(output_overlay_path)

    # git_repo_base = 'C:/Users/mind reader/Desktop/mesonet/mesonet/'
    im = atlas_from_mat(os.path.join(git_repo_base, 'atlases/atlas_512_512.mat'))
    atlas_png = os.path.join(git_repo_base, 'atlases/landmarks_atlas_512_512.png')

    for num, file in enumerate(os.listdir(cwd)):
        if fnmatch.fnmatch(file, "*.png") and 'mask' not in file:
            dlc_img_arr.append(os.path.join(cwd, file))
    for num, file in enumerate(os.listdir(brain_img_dir)):
        if fnmatch.fnmatch(file, "*.png"):
            brain_img_arr.append(os.path.join(brain_img_dir, file))
    i_coord, j_coord = np.array([(100, 256, 413, 256), (148, 254, 148, 446)])
    atlas_arr = np.array([(100, 148), (256, 254), (413, 148), (256, 446)])

    peak_arr_flat = []
    peak_arr_total = []

    if sensory_match:
        for num, file in enumerate(os.listdir(brain_img_dir)):
            sensory_img_for_brain = os.path.join(sensory_img_dir, str(num))
            if glob.glob(sensory_img_for_brain):
                for num_im, file_im in enumerate(os.listdir(sensory_img_for_brain)):
                    sensory_im = io.imread(os.path.join(sensory_img_dir, str(num), file_im))
                    sensory_im = trans.resize(sensory_im, (512, 512))
                    io.imsave(os.path.join(sensory_img_dir, str(num), file_im), sensory_im)
                    peak = find_peaks(os.path.join(sensory_img_dir, str(num), file_im))
                    peak_arr.append(peak)
            for x in peak_arr:
                for y in x:
                    peak_arr_flat.append(y)
            peak_arr_total.append(peak_arr_flat)
            peak_arr_flat = []
            peak_arr = []

    pts = []
    pts2 = []
    pts3 = []
    pts4 = []
    sub_pts = []
    sub_pts2 = []
    sub_pts3 = []
    sub_pts4 = []

    bregma_list = []
    bregma_present = True

    coords = pd.read_csv(coords_input)
    x_coord = coords.iloc[2:, 1::3]
    y_coord = coords.iloc[2:, 2::3]
    landmark_indices = [0, 1, 2, 3] # [0, 3, 2, 1]
    atlas_indices = [0, 1, 2, 3] # [0, 3, 2, 1]
    landmark_indices_hemi = []
    atlas_indices_hemi = []
    one_hemi = False
    two_point = False
    for arr_index, i in enumerate(range(0, len(x_coord))):
        x_coord_flat = x_coord.iloc[i].values.astype('float32')
        y_coord_flat = y_coord.iloc[i].values.astype('float32')
        # x_coord_flat = x_coord_flat[np.r_[0:2, 3:4]]
        # y_coord_flat = y_coord_flat[np.r_[0:2, 3:4]]
        print("landmark arr: {}".format(landmark_arr))
        x_coord_flat = x_coord_flat[landmark_arr]
        y_coord_flat = y_coord_flat[landmark_arr]
        # print(x_coord_flat)
        # print(y_coord_flat)
        # print(len(x_coord_flat))
        # print(len(y_coord_flat))
        dlc_list = []
        atlas_list = []
        mean_dlc = 0
        mean_atlas = 0
        print("Atlas arr: {}".format(atlas_arr))
        for (coord_x, coord_y) in zip(x_coord_flat, y_coord_flat):
            dlc_coord = (coord_x, coord_y)
            # mean_dlc = np.median([coord_x, coord_y])
            # mean_atlas = np.median([coord_i, coord_j])
            dlc_list.append(dlc_coord)
        for coord_atlas in atlas_arr:
            atlas_coord = (coord_atlas[0], coord_atlas[1])
            atlas_list.append(atlas_coord)
            print("coord_atlas: {}".format(coord_atlas))
        # mean_dlc_list.sort()
        # mean_atlas_list.sort()
        a = 0
        b = 0
        # Initialize result as max value
        result = sys.maxsize
        min_landmark_arr = []
        result_atlas_idx = 0

        #for val_dlc, coord_dlc_set in enumerate(mean_dlc_list):
        #    for val_atlas, coord_atlas_set in enumerate(mean_atlas_list):
        #        if abs(coord_dlc_set - coord_atlas_set) < result:
        #            result = abs(coord_dlc_set - coord_atlas_set)
        #            result_atlas_idx = val_atlas
        #    min_landmark_arr.append(result_atlas_idx)
        #landmark_indices = landmark_arr
        #atlas_indices = min_landmark_arr

        print(atlas_list)
        print(dlc_list)
        for val_dlc, coord_dlc_set in enumerate(dlc_list):
            nodes = np.asarray(atlas_list)
            pts_dist = np.sum(abs(nodes - coord_dlc_set), axis=1)
            print("pts dist: {}".format(pts_dist))
            min_dist = np.argmin(pts_dist)
            min_landmark_arr.append(min_dist)
        # landmark_indices = np.argsort(min_landmark_arr).tolist()
        landmark_indices = landmark_indices[0:len(min_landmark_arr)]
        atlas_indices = min_landmark_arr
        print("atlas indices: {}".format(atlas_indices))

        # Adapted from https://www.geeksforgeeks.org/smallest-difference-pair-values-two-unsorted-arrays/

        # Scan Both Arrays up to
        # sizeof of the Arrays
        # while a < len_dlc and b < len_atlas:
        #   if abs(mean_dlc_list[a] - mean_atlas_list[b]) < result:
        #       result = abs(mean_dlc_list[a] - mean_atlas_list[b])
        #        last_a = a
        #        last_b = b
        #        # Move Smaller Value
        #    if (mean_dlc_list[a] < mean_atlas_list[b]):
        #        a += 1
        #    else:
        #        b += 1



        if (len(x_coord_flat) == 3) and (len(y_coord_flat) == 3):
            one_hemi = True
        elif (len(x_coord_flat) == 2) and (len(y_coord_flat) == 2):
            two_point = True
        bregma_index = int(max(landmark_arr))
        if one_hemi:
            if arr_index == 0:
                if 0 in landmark_arr:
                    landmark_indices_hemi.append(0)
                    atlas_indices_hemi.append(0)
                landmark_indices_sorted = sorted(landmark_arr, reverse=True)
                atlas_indices_sorted = sorted(landmark_arr, reverse=True)
                landmark_indices_hemi.extend(landmark_indices_sorted[0:2])
                atlas_indices_hemi.extend(atlas_indices_sorted[0:2])
                # landmark_indices = landmark_indices_hemi
                # atlas_indices = atlas_indices_hemi
            # landmark_indices = [0, 2, 1]
            # atlas_indices = [0, 2, 1]
            # if 1 not in landmark_arr:
            #    atlas_indices = [0, 1, 2]
            bregma_index = int(max(landmark_indices))
            print(bregma_index)
        elif two_point:
            # landmark_indices = [0, 1]
            # atlas_indices = [2, 3]
            bregma_index = int(max(landmark_indices))
            print(bregma_index)
        print(one_hemi)
        print(landmark_indices)
        print(atlas_indices)
        # 0 = left, 1 = bregma, 2 = right, 3 = lambda
        for j in landmark_indices:
            sub_pts.append([x_coord_flat[j], y_coord_flat[j]])
        # 1 = left, 2 = bregma, 3 = right, 0 = lambda
        for j in atlas_indices:
            sub_pts2.append([atlas_arr[j][0], atlas_arr[j][1]])
        pts.append(sub_pts)
        pts2.append(sub_pts2)
        print(sub_pts)
        print(sub_pts2)
        coords_to_mat(sub_pts, i, output_mask_path, bregma_present, bregma_index, landmark_arr)
        sub_pts = []
        sub_pts2 = []

    pts, pts2 = np.asarray(pts).astype('float32'), np.asarray(pts2).astype('float32')
    if sensory_match:
        k_coord, m_coord = np.array([(189, 323, 435, 348), (315, 315, 350, 460)])
        coords_peak = peak_arr_total
        for img_num, img in enumerate(brain_img_arr):
            for j in [1, 0, 3, 2]:  # Get peak values from heatmaps
                sub_pts3.append([coords_peak[img_num][j][0], coords_peak[img_num][j][1]])
            for j in [0, 1, 2, 3]:  # Get circle locations
                sub_pts4.append([k_coord[j], m_coord[j]])
            pts3.append(sub_pts3)
            pts4.append(sub_pts4)
            sensory_to_mat(sub_pts3, pts[img_num][3], img_num, output_mask_path)
            sub_pts3 = []
            sub_pts4 = []
        pts3, pts4 = np.asarray(pts3).astype('float32'), np.asarray(pts4).astype('float32')

    for (n, br), (n_dlc, br_dlc) in zip(enumerate(brain_img_arr), enumerate(dlc_img_arr)):
        io.imsave(os.path.join(cwd, "../output_mask/im.png".format(n)), im)
        cv2.imread(os.path.join(cwd, "../output_mask/im.png".format(n)), cv2.IMREAD_GRAYSCALE)
        atlas_mask_dir = os.path.join(git_repo_base, "atlases/atlas_mask.png")
        atlas_mask = cv2.imread(atlas_mask_dir, cv2.IMREAD_UNCHANGED)
        atlas_mask = cv2.resize(atlas_mask, (im.shape[0], im.shape[1]))
        mask_dir = os.path.join(cwd, "../output_mask/{}.png".format(n))
        print("Performing first transformation of atlas {}...".format(n))
        # First alignment of brain atlas using three cortical landmarks and standard affine transform
        # M = cv2.getAffineTransform(pts2[n][0:3], pts[n][0:3])
        print(pts2[n], pts[n])
        print(len(pts2[n]))
        # homography, mask = cv2.findHomography(pts2[n], pts[n], 0)
        # atlas_warped = warp(im, homography, output_shape=(512, 512))
        # atlas_mask_warped = warp(atlas_mask, homography, output_shape=(512, 512))
        # pts2_for_input = np.delete(np.array(pts2[n]), 1, 0)
        # pts_for_input = np.delete(np.array(pts[n]), 1, 0)
        pts2_for_input = np.array(pts2[n])
        pts_for_input = np.array(pts[n])
        if len(pts2[n]) == 2:
            pts2_for_input = np.append(pts2_for_input, [[0, 0]], axis=0)
            pts_for_input = np.append(pts_for_input, [[0, 0]], axis=0)
        print(pts2_for_input, pts_for_input)
        M = cv2.estimateAffine2D(pts2_for_input, pts_for_input)[0]
        print("M: {}".format(M))
        atlas_warped = cv2.warpAffine(im, M, (512, 512))
        atlas_mask_warped = cv2.warpAffine(atlas_mask, M, (512, 512))
        # homography, mask = cv2.findHomography(pts2[n], pts[n], 0)
        # atlas_align = registerAtlas(br_dlc, atlas_png, cwd, n)
        # atlas_warped = cv2.warpPerspective(im, homography, (512, 512))
        # atlas_warped = warp(im, homography, output_shape=(512, 512))
        # print("M: {}".format(M))
        # atlas_warped = cv2.warpAffine(im, M, (512, 512))
        atlas_first_transform_path = os.path.join(output_mask_path, '{}_atlas_first_transform.png'.format(str(n)))
        io.imsave(atlas_first_transform_path, atlas_warped)
        # atlas_mask_warped = cv2.warpAffine(atlas_mask, M, (512, 512))
        # print("Homography: {}".format(homography))
        # atlas_mask_warped = warp(atlas_mask, homography, output_shape=(512, 512))
        # atlas_mask_warped = cv2.warpPerspective(atlas_mask, homography, (512, 512))
        # Second alignment of brain atlas using four cortical landmarks and piecewise affine transform
        print("Performing second transformation of atlas {}...".format(n))
        dst = getMaskContour(mask_dir, atlas_warped, pts[n], pts2[n], cwd, n, True)
        # If a sensory map of the brain is provided, do a third alignment of the brain atlas using up to four peaks of
        # sensory activity
        if sensory_match:
            dst = getMaskContour(mask_dir, atlas_warped, pts3[n], pts4[n], cwd, n, True)
        # Resize images back to 512x512
        dst = cv2.resize(dst, (im.shape[0], im.shape[1]))
        atlas_mask_warped = getMaskContour(mask_dir, atlas_mask_warped, pts[n], pts2[n], cwd, n, False)
        atlas_mask_warped = cv2.resize(atlas_mask_warped, (im.shape[0], im.shape[1]))
        atlas_path = os.path.join(output_mask_path, '{}_atlas.png'.format(str(n)))
        mask_warped_path = os.path.join(output_mask_path, '{}_mask_warped.png'.format(str(n)))
        io.imsave(atlas_path, dst)
        io.imsave(mask_warped_path, atlas_mask_warped)
        atlas_to_mask(atlas_path, mask_dir, mask_warped_path, output_mask_path, n)
        if bregma_present:
            bregma_list.append(pts[n][bregma_index])
            print("Bregma list: {}".format(bregma_list))
    # Converts the transformed brain atlas into a segmentation method for the original brain image
    applyMask(brain_img_dir, output_mask_path, output_overlay_path, output_overlay_path, mat_save, threshold, git_repo_base,
              bregma_list, region_labels)
