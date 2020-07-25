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
import imutils
import scipy.io
import skimage.io as io
import skimage.transform as trans
from skimage.transform import PiecewiseAffineTransform, warp
import imageio
import os
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


def atlas_from_mat(input_file, mat_cnt_list):
    """
    Generates a binary brain atlas from a .mat file.
    :param input_file: The input .mat file representing a brain atlas (with white = 255 and black = 0)
    :return: A thresholded binary brain atlas.
    """
    file = input_file
    atlas_base = np.zeros((512, 512), dtype="uint8")
    if glob.glob(os.path.join(input_file, '*.mat')):
        mat = scipy.io.loadmat(file)
        mat_shape = mat[list(mat.keys())[3]]
        if len(mat_shape.shape) > 2:
            for val in range(0, mat_shape.shape[2]):
                mat_roi = mat_shape[:, :, val]
                mat_resize = cv2.resize(mat_roi, (512, 512))
                mat_resize = np.uint8(mat_resize)
                ret, thresh = cv2.threshold(mat_resize, 5, 255, cv2.THRESH_BINARY_INV)
                mat_roi_cnt = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                mat_roi_cnt = imutils.grab_contours(mat_roi_cnt)
                c_to_save = max(mat_roi_cnt, key=cv2.contourArea)
                mat_cnt_list.append(c_to_save)
                cv2.drawContours(atlas_base, mat_roi_cnt, -1, (255, 255, 255), 1)
            ret, thresh = cv2.threshold(atlas_base, 5, 255, cv2.THRESH_BINARY_INV)
            io.imsave('atlas_unresized_test.png', thresh)
        else:
            mat = mat['atlas']
            mat_resize = cv2.resize(mat, (512, 512))
            ret, thresh = cv2.threshold(mat_resize, 5, 255, cv2.THRESH_BINARY_INV)
    else:
        atlas_im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # atlas_resize = cv2.resize(atlas_im, (512, 512))
        atlas_resize = np.uint8(atlas_im)
        # atlas_im = cv2.cvtColor(atlas_resize, cv2.COLOR_BGR2GRAY)
        ret, atlas_resize = cv2.threshold(atlas_resize, 127, 255, 0)
        io.imsave('atlas_unresized_test.png', atlas_resize)
        roi_mask_new, roi_cnt, hierarchy = cv2.findContours(atlas_resize, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # roi_cnt = imutils.grab_contours(roi_cnt)
        print("ROI CNT: {}".format(len(roi_cnt)))
        for val in roi_cnt:
            c_to_save = max(val, key=cv2.contourArea)
            mat_cnt_list.append(c_to_save)
            print("MAT CNT: {}".format(len(mat_cnt_list)))
            cv2.drawContours(atlas_base, val, -1, (255, 255, 255), 1)
        ret, thresh = cv2.threshold(atlas_base, 5, 255, cv2.THRESH_BINARY_INV)
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
                    mat_save, threshold, git_repo_base, region_labels, landmark_arr_orig, use_unet,
                    atlas_to_brain_align, model):
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
    mat_cnt_list = []

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
    # im = atlas_from_mat(os.path.join(git_repo_base, 'atlases/atlas_512_512.mat'))
    # im = atlas_from_mat(os.path.join(git_repo_base, 'atlases/atlas.mat'))
    # im = atlas_from_mat(os.path.join(git_repo_base, 'atlases/ROIs_new.mat'), mat_cnt_list)
    if not atlas_to_brain_align:
        # im = atlas_from_mat(os.path.join(git_repo_base, 'atlases/Atlas_workflow2_binary.png'), mat_cnt_list)
        im = cv2.imread(os.path.join(git_repo_base, 'atlases/Atlas_workflow2_binary.png'))
    else:
        im = cv2.imread(os.path.join(git_repo_base, 'atlases/Atlas_workflow1_binary.png'))
        im_left = cv2.imread(os.path.join(git_repo_base, 'atlases/left_hemi.png'))
        ret, im_left = cv2.threshold(im_left, 5, 255, cv2.THRESH_BINARY_INV)
        im_right = cv2.imread(os.path.join(git_repo_base, 'atlases/right_hemi.png'))
        ret, im_right = cv2.threshold(im_right, 5, 255, cv2.THRESH_BINARY_INV)
        im_left = np.uint8(im_left)

        im_right = np.uint8(im_right)
        im = np.uint8(im)
    # im = atlas_from_mat(os.path.join(git_repo_base, 'atlases/atlas_ROIs.mat'))
    atlas = im
    # FOR ALIGNING BRAIN TO ATLAS
    # im_binary = np.uint8(im)

    for num, file in enumerate(os.listdir(cwd)):
        if fnmatch.fnmatch(file, "*.png") and 'mask' not in file:
            dlc_img_arr.append(os.path.join(cwd, file))
    for num, file in enumerate(os.listdir(brain_img_dir)):
        if fnmatch.fnmatch(file, "*.png"):
            brain_img_arr.append(os.path.join(brain_img_dir, file))
        elif fnmatch.fnmatch(file, "*.tif"):
            tif_stack = imageio.mimread(os.path.join(brain_img_dir, file))
            for tif_im in tif_stack:
                brain_img_arr.append(tif_im)
    # i_coord, j_coord = np.array([(100, 256, 413, 256), (148, 254, 148, 446)])
    atlas_arr = np.array([(100.00000, 148.00000), (256.00000, 254.00000), (413.00000, 148.00000),
                          (256.00000, 446.00000)])

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

    bregma_index_list = []
    bregma_list = []
    bregma_present = True

    coords = pd.read_csv(coords_input)
    x_coord = coords.iloc[2:, 1::3]
    y_coord = coords.iloc[2:, 2::3]
    accuracy = coords.iloc[2:, 3::3]
    landmark_indices = [0, 1, 2, 3]  # [0, 3, 2, 1]
    # atlas_indices = [0, 1, 2, 3]  # [0, 3, 2, 1]
    for arr_index, i in enumerate(range(0, len(x_coord))):
        landmark_arr = landmark_arr_orig
        x_coord_flat = x_coord.iloc[i].values.astype('float32')
        y_coord_flat = y_coord.iloc[i].values.astype('float32')
        accuracy_flat = accuracy.iloc[i].values.astype('float32')
        accuracy_where = np.where(accuracy_flat <= 0.20)
        # print("accuracy arr: {}".format(accuracy_where[0]))
        if 0 < (accuracy_where[0]).size < 3:
            landmark_arr_adjusted = np.setdiff1d(np.asarray(landmark_arr), accuracy_where[0]).astype(int)
            landmark_arr = landmark_arr_adjusted
        else:
            print("WARNING: landmarks at positions {} are LOW ACCURACY".format(accuracy_where))
            landmark_arr = landmark_arr_orig
        # print("landmark arr: {}".format(landmark_arr))
        # print("x_coord_flat BEFORE: {}".format(x_coord_flat))
        x_coord_flat = x_coord_flat[landmark_arr]
        # print("x_coord_flat AFTER: {}".format(x_coord_flat))
        y_coord_flat = y_coord_flat[landmark_arr]
        # print(x_coord_flat)
        # print(y_coord_flat)
        # print(len(x_coord_flat))
        # print(len(y_coord_flat))
        dlc_list = []
        atlas_list = []
        print("Atlas arr: {}".format(atlas_arr))
        for (coord_x, coord_y) in zip(x_coord_flat, y_coord_flat):
            dlc_coord = (coord_x, coord_y)
            dlc_list.append(dlc_coord)
        for coord_atlas in atlas_arr:
            atlas_coord = (coord_atlas[0], coord_atlas[1])
            atlas_list.append(atlas_coord)
            # print("coord_atlas: {}".format(coord_atlas))
        # Initialize result as max value
        min_landmark_arr = []

        # print(atlas_list)
        # print(dlc_list)
        for val_dlc, coord_dlc_set in enumerate(dlc_list):
            nodes = np.asarray(atlas_list)
            pts_dist = np.sum(abs(nodes - coord_dlc_set), axis=1)
            # print("pts dist: {}".format(pts_dist))
            min_dist = np.argmin(pts_dist)
            min_landmark_arr.append(min_dist)
        # landmark_indices = np.argsort(min_landmark_arr).tolist()
        landmark_indices = landmark_indices[0:len(landmark_arr)]
        atlas_indices = min_landmark_arr
        print("atlas indices: {}".format(atlas_indices))

        pts_dist = np.absolute(np.asarray(atlas_list) - np.asarray((im.shape[0]/2, im.shape[1]/2)))
        pts_avg_dist = [np.mean(v) for v in pts_dist]
        # print("bregma dist: {}".format(pts_avg_dist))
        bregma_index = np.argmin(np.asarray(pts_avg_dist))
        # print("bregma index: {}".format(bregma_index))

        # 0 = left, 1 = bregma, 2 = right, 3 = lambda
        for j in landmark_indices:
            sub_pts.append([x_coord_flat[j], y_coord_flat[j]])
            # print("sub_pts: {}".format(sub_pts))
        # 1 = left, 2 = bregma, 3 = right, 0 = lambda
        for j in atlas_indices:
            sub_pts2.append([atlas_arr[j][0], atlas_arr[j][1]])
            # print("sub_pts2: {}".format(sub_pts2))
        # print(sub_pts)
        # print(sub_pts2)
        pts.append(sub_pts)
        pts2.append(sub_pts2)
        coords_to_mat(sub_pts, i, output_mask_path, bregma_present, bregma_index, landmark_arr)
        print(bregma_index)
        bregma_index_list.append(bregma_index)
        sub_pts = []
        sub_pts2 = []
    # pts = np.array(pts).astype('float32')
    # print("pts: {}".format(pts))
    # pts2 = np.array(pts2).astype('float32')
    # print("pts2: {}".format(pts2))
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

    for (n, br) in enumerate(brain_img_arr):
        if atlas_to_brain_align:
            im = np.uint8(im)
        else:
            # FOR ALIGNING BRAIN TO ATLAS
            if '.png' in br:
                im = cv2.imread(br)
            else:
                im = br
            im = np.uint8(im)
            im = cv2.resize(im, (512, 512))

        io.imsave(os.path.join(cwd, "../output_mask/atlas_unskewed.png".format(n)), im)
        # cv2.imread(os.path.join(cwd, "../output_mask/atlas_unskewed.png".format(n)), cv2.IMREAD_GRAYSCALE)
        if atlas_to_brain_align:
            atlas_mask_dir = os.path.join(git_repo_base, "atlases/Atlas_workflow1_smooth_binary.png")
            atlas_mask_dir_left = os.path.join(git_repo_base, "atlases/left_hemisphere_smooth.png")
            atlas_mask_dir_right = os.path.join(git_repo_base, "atlases/right_hemisphere_smooth.png")
        else:
            atlas_mask_dir = os.path.join(git_repo_base, "atlases/Atlas_workflow2_smooth_binary.png")
        atlas_mask = cv2.imread(atlas_mask_dir, cv2.IMREAD_UNCHANGED)
        atlas_mask = cv2.resize(atlas_mask, (im.shape[0], im.shape[1]))
        atlas_mask = np.uint8(atlas_mask)
        atlas_mask_left = cv2.imread(atlas_mask_dir_left, cv2.IMREAD_UNCHANGED)
        atlas_mask_left = cv2.resize(atlas_mask_left, (im.shape[0], im.shape[1]))
        atlas_mask_left = np.uint8(atlas_mask_left)
        atlas_mask_right = cv2.imread(atlas_mask_dir_right, cv2.IMREAD_UNCHANGED)
        atlas_mask_right = cv2.resize(atlas_mask_right, (im.shape[0], im.shape[1]))
        atlas_mask_right = np.uint8(atlas_mask_right)
        mask_dir = os.path.join(cwd, "../output_mask/{}.png".format(n))
        print("Performing first transformation of atlas {}...".format(n))
        # First alignment of brain atlas using three cortical landmarks and standard affine transform
        # warp_coords = cv2.getAffineTransform(pts2[n][0:3], pts[n][0:3])
        # print(pts2[n], pts[n])
        # print(len(pts2[n]))
        # homography, mask = cv2.findHomography(pts2[n], pts[n], 0)
        # atlas_warped = warp(im, homography, output_shape=(512, 512))
        # atlas_mask_warped = warp(atlas_mask, homography, output_shape=(512, 512))
        # pts2_for_input = np.delete(np.array(pts2[n]), 1, 0)
        # pts_for_input = np.delete(np.array(pts[n]), 1, 0)
        pts2_for_input = np.array(pts2[n][0:len(pts[n])]).astype('float32')
        pts_for_input = np.array(pts[n]).astype('float32')
        # print(pts2_for_input)
        # print(pts_for_input)
        if len(pts2_for_input) == 2:
            pts2_for_input = np.append(pts2_for_input, [[0, 0]], axis=0)
            pts_for_input = np.append(pts_for_input, [[0, 0]], axis=0)
        if len(pts2_for_input) <= 2:
            warp_coords = cv2.estimateAffinePartial2D(pts2_for_input, pts_for_input)[0]
            atlas_warped = cv2.warpAffine(im, warp_coords, (512, 512))
        elif len(pts2_for_input) == 3:
            warp_coords = cv2.getAffineTransform(pts2_for_input, pts_for_input)
            atlas_warped = cv2.warpAffine(im, warp_coords, (512, 512))
        elif len(pts2_for_input) == 4:
            if atlas_to_brain_align:
                # pts2_left = np.reshape(np.take(pts2[n], [0, 2, 3], axis=0), (3, 2))
                # pts2_right = np.reshape(np.take(pts2[n], [1, 2, 3], axis=0), (3, 2))
                # pts1_left = np.reshape(np.take(pts[n], [0, 2, 3], axis=0), (3, 2))
                # pts1_right = np.reshape(np.take(pts[n], [1, 2, 3], axis=0), (3, 2))
                print(np.array(pts2[n], dtype=np.float32))
                pts2_left = np.array([pts2[n][0], pts2[n][2], pts2[n][3]], dtype=np.float32)
                pts2_right = np.array([pts2[n][1], pts2[n][2], pts2[n][3]], dtype=np.float32)
                pts1_left = np.array([pts[n][0], pts[n][2], pts[n][3]], dtype=np.float32)
                pts1_right = np.array([pts[n][1], pts[n][2], pts[n][3]], dtype=np.float32)
                print(pts2_left.flags)
                print(pts1_left.flags)
                print("WARP COORDS: {}, {}".format(pts2_left, pts1_left))
                warp_coords_left = cv2.getAffineTransform(pts2_left, pts1_left)
                warp_coords_right = cv2.getAffineTransform(pts2_right, pts1_right)
                atlas_warped_left = cv2.warpAffine(im_left, warp_coords_left, (512, 512))
                atlas_warped_right = cv2.warpAffine(im_right, warp_coords_right, (512, 512))
                atlas_warped = cv2.bitwise_or(atlas_warped_left, atlas_warped_right)
                ret, atlas_warped = cv2.threshold(atlas_warped, 5, 255, cv2.THRESH_BINARY_INV)
                atlas_left_transform_path = os.path.join(output_mask_path,
                                                          '{}_atlas_left_transform.png'.format(str(n)))
                atlas_right_transform_path = os.path.join(output_mask_path,
                                                          '{}_atlas_right_transform.png'.format(str(n)))
                io.imsave(atlas_left_transform_path, atlas_warped_left)
                io.imsave(atlas_right_transform_path, atlas_warped_right)
                # warp_coords = cv2.getAffineTransform(pts2_for_input, pts_for_input)
                # warp_coords = cv2.getPerspectiveTransform(pts2_for_input, pts_for_input)
                # atlas_warped = cv2.warpPerspective(im, warp_coords, (512, 512))
            else:
                warp_coords = cv2.getAffineTransform(pts2_for_input, pts_for_input)
                atlas_warped = cv2.warpAffine(im, warp_coords, (512, 512))
        # print("warp_coords: {}".format(warp_coords))

        # COMMENT OUT FOR ALIGNING BRAIN TO ATLAS
        if atlas_to_brain_align:
            if len(pts2_for_input) <= 4:
                # atlas_mask_warped = cv2.warpAffine(atlas_mask, warp_coords, (512, 512))
                atlas_mask_left_warped = cv2.warpAffine(atlas_mask_left, warp_coords_left, (512, 512))
                atlas_mask_right_warped = cv2.warpAffine(atlas_mask_right, warp_coords_right, (512, 512))
                atlas_mask_warped = cv2.bitwise_or(atlas_mask_left_warped, atlas_mask_right_warped)
                # atlas_mask_warped = atlas_mask
            # elif len(pts2_for_input) == 4:
            #    atlas_mask_warped = cv2.warpPerspective(atlas_mask, warp_coords, (512, 512))

        atlas_first_transform_path = os.path.join(output_mask_path, '{}_atlas_first_transform.png'.format(str(n)))
        io.imsave(atlas_first_transform_path, atlas_warped)
        # Second alignment of brain atlas using cortical landmarks and piecewise affine transform
        print("Performing second transformation of atlas {}...".format(n))
        if use_unet == 1 and atlas_to_brain_align:
            dst = getMaskContour(mask_dir, atlas_warped, pts[n], pts2[n], cwd, n, True)
        else:
            dst = atlas_warped
        mask_warped_path = os.path.join(output_mask_path, '{}_mask_warped.png'.format(str(n)))
        # If a sensory map of the brain is provided, do a third alignment of the brain atlas using up to four peaks of
        # sensory activity
        if sensory_match:
            if use_unet == 1:
                dst = getMaskContour(mask_dir, atlas_warped, pts3[n], pts4[n], cwd, n, True)
            else:
                warp_sensory_coords = cv2.estimateAffine2D(pts4[n], pts3[n])[0]
                dst = cv2.warpAffine(atlas_warped, warp_sensory_coords, (512, 512))
                # COMMENT OUT FOR ALIGNING BRAIN TO ATLAS
                if atlas_to_brain_align:
                    atlas_mask_warped = cv2.warpAffine(atlas_mask_warped, warp_sensory_coords, (512, 512))
                    atlas_mask_warped = getMaskContour(mask_dir, atlas_mask_warped, pts[n], pts2[n], cwd, n, False)
                    atlas_mask_warped = cv2.resize(atlas_mask_warped, (im.shape[0], im.shape[1]))
        if atlas_to_brain_align:
            io.imsave(mask_warped_path, atlas_mask_warped)
        else:
            io.imsave(mask_warped_path, atlas_mask)
        # Resize images back to 512x512
        dst = cv2.resize(dst, (im.shape[0], im.shape[1]))
        atlas_path = os.path.join(output_mask_path, '{}_atlas.png'.format(str(n)))
        if atlas_to_brain_align:
            io.imsave(atlas_path, dst)
        else:
            brain_warped_path = os.path.join(output_mask_path, '{}_brain_warp.png'.format(str(n)))
            # for pt in pts[n]:
            #     print((pt[0], pt[1]))
            #     cv2.circle(dst, (int(pt[0]), int(pt[1])), 10, (255, 255, 255), -1)
            io.imsave(brain_warped_path, dst)
            io.imsave(atlas_path, atlas)
        if atlas_to_brain_align:
            atlas_to_mask(atlas_path, mask_dir, mask_warped_path, output_mask_path, n, use_unet,
                          atlas_to_brain_align)
        if bregma_present:
            bregma_val = int(bregma_index_list[n])
            bregma_list.append(pts[n][bregma_val])
            print("Bregma list: {}".format(bregma_list))
    # Converts the transformed brain atlas into a segmentation method for the original brain image
    print(bregma_list)
    applyMask(brain_img_dir, output_mask_path, output_overlay_path, output_overlay_path, mat_save, threshold,
              git_repo_base, bregma_list, atlas_to_brain_align, model, mat_cnt_list, pts, region_labels)
