"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
from mesonet.utils import natural_sort_key, convert_to_png
from mesonet.mask_functions import atlas_to_mask, applyMask

# from mesonet.niftyreg_align import niftyreg_align
from mesonet.voxelmorph_align import voxelmorph_align, vxm_transform
import numpy as np
import pandas as pd
import cv2
import imutils
import math
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
    max_loc_arr = []
    img = cv2.imread(str(img), 0)
    im = img.copy()
    x_min = int(np.around(im.shape[0] / 2))
    im1 = im[:, x_min : im.shape[0]]
    im2 = im[:, 0:x_min]
    (minVal, max_val, minLoc, max_loc) = cv2.minMaxLoc(im1)
    (minVal2, maxVal2, minLoc2, maxLoc2) = cv2.minMaxLoc(im2)
    max_loc = list(max_loc)
    max_loc[0] = max_loc[0] + x_min
    max_loc = tuple(max_loc)
    max_loc_arr.append(max_loc)
    if (max_val - 30) <= maxVal2 <= (max_val + 30):
        max_loc_arr.append(maxLoc2)
    return max_loc_arr


def coords_to_mat(
    sub_dlc_pts, i, output_mask_path, bregma_present, bregma_index, landmark_arr
):
    if bregma_present:
        x_bregma, y_bregma = sub_dlc_pts[bregma_index]
        for pt, landmark in zip(sub_dlc_pts, landmark_arr):
            x_pt = pt[0]
            y_pt = pt[1]
            pt_adj = [landmark, x_pt - x_bregma, y_pt - y_bregma]
            pt_adj_to_mat = np.array(pt_adj, dtype=np.object)
            if not os.path.isdir(os.path.join(output_mask_path, "mat_coords")):
                os.mkdir(os.path.join(output_mask_path, "mat_coords"))
            scipy.io.savemat(
                os.path.join(
                    output_mask_path,
                    "mat_coords/landmarks_{}_{}.mat".format(i, landmark),
                ),
                {"landmark_coords_{}_{}".format(i, landmark): pt_adj_to_mat},
                appendmat=False,
            )


def sensory_to_mat(sub_dlc_pts, bregma_pt, i, output_mask_path):
    x_bregma, y_bregma = bregma_pt
    sensory_names = ["tail_left", "tail_right", "visual", "whisker"]
    for pt, landmark in zip(sub_dlc_pts, sensory_names):
        x_pt = pt[0]
        y_pt = pt[1]
        pt_adj = [landmark, x_pt - x_bregma, y_pt - y_bregma]
        pt_adj_to_mat = np.array(pt_adj, dtype=np.object)
        if not os.path.isdir(os.path.join(output_mask_path, "mat_coords")):
            os.mkdir(os.path.join(output_mask_path, "mat_coords"))
        scipy.io.savemat(
            os.path.join(
                output_mask_path,
                "mat_coords/sensory_peaks_{}_{}.mat".format(i, landmark),
            ),
            {"sensory_peaks_{}_{}".format(i, landmark): pt_adj_to_mat},
            appendmat=False,
        )


def atlas_from_mat(input_file, mat_cnt_list):
    """
    Generates a binary brain atlas from a .mat file.
    :param input_file: The input .mat file representing a brain atlas (with white = 255 and black = 0)
    :param mat_cnt_list: The list to which mat files should be appended
    :return: A thresholded binary brain atlas.
    """
    file = input_file
    atlas_base = np.zeros((512, 512), dtype="uint8")
    if glob.glob(os.path.join(input_file, "*.mat")):
        mat = scipy.io.loadmat(file)
        mat_shape = mat[list(mat.keys())[3]]
        if len(mat_shape.shape) > 2:
            for val in range(0, mat_shape.shape[2]):
                mat_roi = mat_shape[:, :, val]
                mat_resize = cv2.resize(mat_roi, (512, 512))
                mat_resize = np.uint8(mat_resize)
                ret, thresh = cv2.threshold(mat_resize, 5, 255, cv2.THRESH_BINARY_INV)
                mat_roi_cnt = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                mat_roi_cnt = imutils.grab_contours(mat_roi_cnt)
                c_to_save = max(mat_roi_cnt, key=cv2.contourArea)
                mat_cnt_list.append(c_to_save)
                cv2.drawContours(atlas_base, mat_roi_cnt, -1, (255, 255, 255), 1)
            ret, thresh = cv2.threshold(atlas_base, 5, 255, cv2.THRESH_BINARY_INV)
            io.imsave("atlas_unresized_test.png", thresh)
        else:
            mat = mat["atlas"]
            mat_resize = cv2.resize(mat, (512, 512))
            ret, thresh = cv2.threshold(mat_resize, 5, 255, cv2.THRESH_BINARY_INV)
    else:
        atlas_im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        atlas_resize = np.uint8(atlas_im)
        ret, atlas_resize = cv2.threshold(atlas_resize, 127, 255, 0)
        io.imsave("atlas_unresized_test.png", atlas_resize)
        roi_cnt, hierarchy = cv2.findContours(
            atlas_resize, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        for val in roi_cnt:
            c_to_save = max(val, key=cv2.contourArea)
            mat_cnt_list.append(c_to_save)
            cv2.drawContours(atlas_base, val, -1, (255, 255, 255), 1)
        ret, thresh = cv2.threshold(atlas_base, 5, 255, cv2.THRESH_BINARY_INV)
    return thresh


def atlas_rotate(dlc_pts, im):
    dlc_y_pts = [
        coord if (190 <= coord[0] <= 330) else (1000, 1000) for coord in dlc_pts
    ]
    dlc_y_pts = [coord for coord in dlc_y_pts if coord[0] < 1000]

    # https://stackoverflow.com/questions/42258637/how-to-know-the-angle-between-two-points
    rotate_rad = math.atan2(0, (im.shape[1] / 2) - dlc_y_pts[-1][0])
    rotate_deg = -1 * (abs(math.degrees(rotate_rad)))
    im_rotate_mat = cv2.getRotationMatrix2D(
        (im.shape[1] / 2, im.shape[0] / 2), rotate_deg, 1.0
    )
    im_rotated = cv2.warpAffine(im, im_rotate_mat, (512, 512))
    x_min = int(np.around(im_rotated.shape[0] / 2))
    im_left = im_rotated[:, 0:x_min]
    im_right = im_rotated[:, x_min : im_rotated.shape[0]]
    return im_left, im_right


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
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        cnt = cnt[:, 0, :]
        cnt = np.asarray(cnt).astype("float32")
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


def homography_match(warp_from, warp_to, output_mask_path, n):
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(warp_from, None)  # im
    kp2, d2 = orb_detector.detectAndCompute(warp_to, None)  # atlas_warped

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[: int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    atlas_warped = cv2.warpPerspective(warp_to, homography, (512, 512))
    atlas_homography_transform_path = os.path.join(
        output_mask_path, "{}_atlas_homography.png".format(str(n))
    )
    io.imsave(atlas_homography_transform_path, atlas_warped)
    return atlas_warped


def atlasBrainMatch(
    brain_img_dir,
    sensory_img_dir,
    coords_input,
    sensory_match,
    mat_save,
    threshold,
    git_repo_base,
    region_labels,
    landmark_arr_orig,
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
    voxelmorph_model="motif_model_atlas.h5",
    template_path="templates",
    flow_path="",
):
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
    :param landmark_arr_orig: The original array of landmarks from DeepLabCut (to be distinguished from any automatic
    exclusions to landmark array based on prediction quality).
    :param use_unet: Choose whether or not to identify the borders of the cortex using a U-net model.
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
    :param exist_transform: if True, uses an existing voxelmorph transformation field for all data instead of predicting
    a new transformation.
    :param voxelmorph_model: the name of a .h5 model located in the models folder of the git repository for MesoNet,
    generated using voxelmorph and containing weights for a voxelmorph local deformation model.
    :param template_path: the path to a template atlas (.npy or .mat( to which the brain image will be aligned in
    voxelmorph.
    :param flow_path: the path to a voxelmorph transformation field that will be used to transform all data instead of
    predicting a new transformation if exist_transform is True.
    """
    # load brain images folder
    brain_img_arr = []
    dlc_img_arr = []
    peak_arr = []
    atlas_label_list = []

    # Prepare template for voxelmorph
    convert_to_png(template_path)
    template = cv2.imread(
        glob.glob(os.path.join(git_repo_base, "atlases", template_path, "*.png"))[0]
    )
    template = np.uint8(template)
    template = cv2.resize(template, (512, 512))

    # Prepare output folder
    cwd = os.getcwd()
    output_mask_path = os.path.join(cwd, "../output_mask")
    # Output folder for transparent masks and masks overlaid onto brain image
    output_overlay_path = os.path.join(cwd, "../output_overlay")
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    if not os.path.isdir(output_overlay_path):
        os.mkdir(output_overlay_path)

    if not atlas_to_brain_align:
        im = cv2.imread(
            os.path.join(git_repo_base, "atlases/Atlas_workflow2_binary.png")
        )
    else:
        if use_voxelmorph and not use_dlc:
            im = cv2.imread(
                os.path.join(git_repo_base, "atlases/Atlas_for_Voxelmorph_binary.png")
            )
            print('ATLAS FOR VXM BINARY')
        else:
            im = cv2.imread(
                os.path.join(git_repo_base, "atlases/Atlas_workflow1_binary.png")
            )
        im_left = cv2.imread(os.path.join(git_repo_base, "atlases/left_hemi.png"))
        ret, im_left = cv2.threshold(im_left, 5, 255, cv2.THRESH_BINARY_INV)
        im_right = cv2.imread(os.path.join(git_repo_base, "atlases/right_hemi.png"))
        ret, im_right = cv2.threshold(im_right, 5, 255, cv2.THRESH_BINARY_INV)
        im_left = np.uint8(im_left)
        im_right = np.uint8(im_right)
        im = np.uint8(im)
    # im = atlas_from_mat(os.path.join(git_repo_base, 'atlases/atlas_ROIs.mat'))
    atlas = im
    # FOR ALIGNING BRAIN TO ATLAS
    # im_binary = np.uint8(im)

    for num, file in enumerate(os.listdir(cwd)):
        if fnmatch.fnmatch(file, "*.png") and "mask" not in file:
            dlc_img_arr.append(os.path.join(cwd, file))
    for num, file in enumerate(os.listdir(brain_img_dir)):
        if fnmatch.fnmatch(file, "*.png"):
            brain_img_arr.append(os.path.join(brain_img_dir, file))
            brain_img_arr.sort(key=natural_sort_key)
        elif fnmatch.fnmatch(file, "*.tif"):
            tif_stack = imageio.mimread(os.path.join(brain_img_dir, file))
            for tif_im in tif_stack:
                brain_img_arr.append(tif_im)
    # i_coord, j_coord = np.array([(100, 256, 413, 256), (148, 254, 148, 446)])
    # https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    coord_circles_img = cv2.imread(
        os.path.join(
            git_repo_base, "atlases", "multi_landmark", "landmarks_new_binary.png"
        ),
        cv2.IMREAD_GRAYSCALE,
    )
    coord_circles_img = np.uint8(coord_circles_img)
    # detect circles in the image
    circles, hierarchy = cv2.findContours(
        coord_circles_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        atlas_arr = np.array(
            [
                (
                    int(cv2.moments(circle)["m10"] / cv2.moments(circle)["m00"]),
                    int(cv2.moments(circle)["m01"] / cv2.moments(circle)["m00"]),
                )
                for circle in circles
            ]
        )

    atlas_arr = np.array(
        [
            (102, 148),
            (166, 88),
            (214, 454),
            (256, 88),
            (256, 256),
            (256, 428),
            (410, 148),
            (346, 88),
            (298, 454),
        ]
    )

    peak_arr_flat = []
    peak_arr_total = []

    if sensory_match:
        for num, file in enumerate(os.listdir(brain_img_dir)):
            sensory_img_for_brain = os.path.join(sensory_img_dir, str(num))
            if glob.glob(sensory_img_for_brain):
                for num_im, file_im in enumerate(os.listdir(sensory_img_for_brain)):
                    sensory_im = io.imread(
                        os.path.join(sensory_img_dir, str(num), file_im)
                    )
                    sensory_im = trans.resize(sensory_im, (512, 512))
                    io.imsave(
                        os.path.join(sensory_img_dir, str(num), file_im), sensory_im
                    )
                    peak = find_peaks(os.path.join(sensory_img_dir, str(num), file_im))
                    peak_arr.append(peak)
            for x in peak_arr:
                for y in x:
                    peak_arr_flat.append(y)
            peak_arr_total.append(peak_arr_flat)
            peak_arr_flat = []
            peak_arr = []

    dlc_pts = []
    atlas_pts = []
    sensory_peak_pts = []
    sensory_atlas_pts = []
    sub_dlc_pts = []
    sub_atlas_pts = []
    sub_sensory_peak_pts = []
    sub_sensory_atlas_pts = []

    bregma_index_list = []
    bregma_list = []
    if use_dlc:
        bregma_present = True
    else:
        bregma_present = False

    coords = pd.read_csv(coords_input)
    x_coord = coords.iloc[2:, 1::3]
    y_coord = coords.iloc[2:, 2::3]
    accuracy = coords.iloc[2:, 3::3]
    acc_left_total = accuracy.iloc[:, 0:5]
    acc_right_total = accuracy.iloc[:, 3:8]
    landmark_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # [0, 3, 2, 1]
    for arr_index, i in enumerate(range(0, len(x_coord))):
        landmark_arr = landmark_arr_orig
        x_coord_flat = x_coord.iloc[i].values.astype("float32")
        y_coord_flat = y_coord.iloc[i].values.astype("float32")
        x_coord_flat = x_coord_flat[landmark_arr]
        y_coord_flat = y_coord_flat[landmark_arr]
        dlc_list = []
        atlas_list = []
        for (coord_x, coord_y) in zip(x_coord_flat, y_coord_flat):
            dlc_coord = (coord_x, coord_y)
            dlc_list.append(dlc_coord)
        for coord_atlas in atlas_arr:
            atlas_coord = (coord_atlas[0], coord_atlas[1])
            atlas_list.append(atlas_coord)
        atlas_list = [atlas_list[i] for i in landmark_arr]
        # Initialize result as max value

        landmark_indices = landmark_indices[0 : len(landmark_arr)]

        # atlas_indices = min_landmark_arr
        atlas_indices = landmark_arr

        # print('atlas indices: {}'.format(atlas_indices))
        # print('landmark indices: {}'.format(landmark_indices))
        # print('x coords: {}'.format(x_coord_flat))

        pts_dist = np.absolute(
            np.asarray(atlas_list) - np.asarray((im.shape[0] / 2, im.shape[1] / 2))
        )
        pts_avg_dist = [np.mean(v) for v in pts_dist]
        # print("bregma dist: {}".format(pts_avg_dist))
        bregma_index = np.argmin(np.asarray(pts_avg_dist))
        # print("bregma index: {}".format(bregma_index))

        for j in landmark_indices:
            sub_dlc_pts.append([x_coord_flat[j], y_coord_flat[j]])
        for j in atlas_indices:
            sub_atlas_pts.append([atlas_arr[j][0], atlas_arr[j][1]])

        dlc_pts.append(sub_dlc_pts)
        atlas_pts.append(sub_atlas_pts)
        coords_to_mat(
            sub_dlc_pts, i, output_mask_path, bregma_present, bregma_index, landmark_arr
        )
        bregma_index_list.append(bregma_index)
        sub_dlc_pts = []
        sub_atlas_pts = []
    if sensory_match:
        k_coord, m_coord = np.array([(189, 323, 435, 348), (315, 315, 350, 460)])
        coords_peak = peak_arr_total
        for img_num, img in enumerate(brain_img_arr):
            for j in [1, 0, 3, 2]:  # Get peak values from heatmaps
                sub_sensory_peak_pts.append(
                    [coords_peak[img_num][j][0], coords_peak[img_num][j][1]]
                )
            for j in [0, 1, 2, 3]:  # Get circle locations
                sub_sensory_atlas_pts.append([k_coord[j], m_coord[j]])
            sensory_peak_pts.append(sub_sensory_peak_pts)
            sensory_atlas_pts.append(sub_sensory_atlas_pts)
            sensory_to_mat(
                sub_sensory_peak_pts, dlc_pts[img_num][3], img_num, output_mask_path
            )
            sub_sensory_peak_pts = []
            sub_sensory_atlas_pts = []
        sensory_peak_pts, sensory_atlas_pts = (
            np.asarray(sensory_peak_pts).astype("float32"),
            np.asarray(sensory_atlas_pts).astype("float32"),
        )

    for (n, br) in enumerate(brain_img_arr):
        align_val = n
        if atlas_to_brain_align:
            im = np.uint8(im)
            br = cv2.imread(br)
            br_vxm = cv2.resize(br, (512, 512))
            br_vxm = np.uint8(br_vxm)
        else:
            # FOR ALIGNING BRAIN TO ATLAS
            if ".png" in br:
                im = cv2.imread(br)
            else:
                im = br
            im = np.uint8(im)
            im = cv2.resize(im, (512, 512))

        if atlas_to_brain_align:
            # atlas_mask_dir = os.path.join(git_repo_base, "atlases/Atlas_workflow1_smooth_binary.png")
            if use_voxelmorph and not use_dlc:
                atlas_mask_dir = os.path.join(
                    git_repo_base, "atlases/Atlas_for_Voxelmorph_border.png"
                )
            else:
                atlas_mask_dir = os.path.join(
                    git_repo_base, "atlases/atlas_smooth2_binary.png"
                )
            atlas_mask_dir_left = os.path.join(
                git_repo_base, "atlases/left_hemisphere_smooth.png"
            )
            atlas_mask_dir_right = os.path.join(
                git_repo_base, "atlases/right_hemisphere_smooth.png"
            )
            atlas_label_mask_dir = os.path.join(
                git_repo_base, "atlases/diff_colour_regions/Common_atlas.mat"
            )
            atlas_label_mask_dir_left = os.path.join(
                git_repo_base, "atlases/diff_colour_regions/atlas_left_hemisphere.csv"
            )
            atlas_label_mask_dir_right = os.path.join(
                git_repo_base, "atlases/diff_colour_regions/atlas_right_hemisphere.csv"
            )
            # atlas_label_mask_left = atlas_from_mat(atlas_label_mask_dir_left, [])
            # atlas_label_mask_right = atlas_from_mat(atlas_label_mask_dir_right, [])
            atlas_label_mask_left = np.genfromtxt(
                atlas_label_mask_dir_left, delimiter=","
            )
            atlas_label_mask_right = np.genfromtxt(
                atlas_label_mask_dir_right, delimiter=","
            )
            atlas_mask_left = cv2.imread(atlas_mask_dir_left, cv2.IMREAD_UNCHANGED)
            atlas_mask_left = cv2.resize(atlas_mask_left, (im.shape[0], im.shape[1]))
            atlas_mask_left = np.uint8(atlas_mask_left)
            atlas_mask_right = cv2.imread(atlas_mask_dir_right, cv2.IMREAD_UNCHANGED)
            atlas_mask_right = cv2.resize(atlas_mask_right, (im.shape[0], im.shape[1]))
            atlas_mask_right = np.uint8(atlas_mask_right)
        else:
            atlas_mask_dir = os.path.join(
                git_repo_base, "atlases/atlas_smooth2_binary.png"
            )
        atlas_mask = cv2.imread(atlas_mask_dir, cv2.IMREAD_UNCHANGED)
        atlas_mask = cv2.resize(atlas_mask, (im.shape[0], im.shape[1]))
        atlas_mask = np.uint8(atlas_mask)
        mask_dir = os.path.join(cwd, "../output_mask/{}.png".format(n))

        print("Performing first transformation of atlas {}...".format(n))

        mask_warped_path = os.path.join(
            output_mask_path, "{}_mask_warped.png".format(str(n))
        )
        if use_dlc:
            # First alignment of brain atlas using three cortical landmarks and standard affine transform
            atlas_pts_for_input = np.array([atlas_pts[n][0 : len(dlc_pts[n])]]).astype(
                "float32"
            )
            pts_for_input = np.array([dlc_pts[n]]).astype("float32")

            if align_once:
                align_val = 0
            else:
                align_val = n

            if len(atlas_pts_for_input[0]) == 2:
                atlas_pts_for_input = np.append(atlas_pts_for_input[0], [[0, 0]], axis=0)
                pts_for_input = np.append(pts_for_input[0], [[0, 0]], axis=0)
            if len(atlas_pts_for_input[0]) <= 2:
                warp_coords = cv2.estimateAffinePartial2D(
                    atlas_pts_for_input, pts_for_input
                )[0]
                if atlas_to_brain_align:
                    atlas_warped_left = cv2.warpAffine(im_left, warp_coords, (512, 512))
                    atlas_warped_right = cv2.warpAffine(im_right, warp_coords, (512, 512))
                    atlas_warped = cv2.bitwise_or(atlas_warped_left, atlas_warped_right)
                    ret, atlas_warped = cv2.threshold(
                        atlas_warped, 5, 255, cv2.THRESH_BINARY_INV
                    )
                    atlas_left_transform_path = os.path.join(
                        output_mask_path, "{}_atlas_left_transform.png".format(str(n))
                    )
                    atlas_right_transform_path = os.path.join(
                        output_mask_path, "{}_atlas_right_transform.png".format(str(n))
                    )
                    io.imsave(atlas_left_transform_path, atlas_warped_left)
                    io.imsave(atlas_right_transform_path, atlas_warped_right)
                else:
                    atlas_warped = cv2.warpAffine(im, warp_coords, (512, 512))
            elif len(atlas_pts_for_input[0]) == 3:
                warp_coords = cv2.getAffineTransform(atlas_pts_for_input, pts_for_input)
                atlas_warped = cv2.warpAffine(im, warp_coords, (512, 512))
            elif len(atlas_pts_for_input[0]) >= 4:
                im_final_size = (512, 512)

                left = acc_left_total.iloc[n, :].values.astype("float32").tolist()
                right = acc_right_total.iloc[n, :].values.astype("float32").tolist()
                left = np.argsort(left).tolist()
                right = np.argsort(right).tolist()
                right = [x + 1 for x in right]
                if set([1, 3, 5, 7]).issubset(landmark_arr):
                    left = [1, 3, 5]
                    right = [3, 5, 7]
                else:
                    left = [x for x in landmark_indices if x in range(0, 6)][0:2]
                    right = [x for x in landmark_indices if x in range(3, 9)][0:2]

                try:
                    atlas_pts_left = np.array(
                        [
                            atlas_pts[align_val][left[0]],
                            atlas_pts[align_val][left[1]],
                            atlas_pts[align_val][left[2]],
                        ],
                        dtype=np.float32,
                    )
                    atlas_pts_right = np.array(
                        [
                            atlas_pts[align_val][right[0]],
                            atlas_pts[align_val][right[1]],
                            atlas_pts[align_val][right[2]],
                        ],
                        dtype=np.float32,
                    )
                    dlc_pts_left = np.array(
                        [
                            dlc_pts[align_val][left[0]],
                            dlc_pts[align_val][left[1]],
                            dlc_pts[align_val][left[2]],
                        ],
                        dtype=np.float32,
                    )
                    dlc_pts_right = np.array(
                        [
                            dlc_pts[align_val][right[0]],
                            dlc_pts[align_val][right[1]],
                            dlc_pts[align_val][right[2]],
                        ],
                        dtype=np.float32,
                    )

                except:
                    atlas_pts_left = np.array(
                        [
                            atlas_pts[align_val][0],
                            atlas_pts[align_val][2],
                            atlas_pts[align_val][3],
                        ],
                        dtype=np.float32,
                    )
                    atlas_pts_right = np.array(
                        [
                            atlas_pts[align_val][1],
                            atlas_pts[align_val][2],
                            atlas_pts[align_val][3],
                        ],
                        dtype=np.float32,
                    )
                    dlc_pts_left = np.array(
                        [
                            dlc_pts[align_val][0],
                            dlc_pts[align_val][2],
                            dlc_pts[align_val][3],
                        ],
                        dtype=np.float32,
                    )
                    dlc_pts_right = np.array(
                        [
                            dlc_pts[align_val][1],
                            dlc_pts[align_val][2],
                            dlc_pts[align_val][3],
                        ],
                        dtype=np.float32,
                    )

                warp_coords_left = cv2.getAffineTransform(atlas_pts_left, dlc_pts_left)
                warp_coords_right = cv2.getAffineTransform(atlas_pts_right, dlc_pts_right)
                if atlas_to_brain_align:
                    atlas_warped_left = cv2.warpAffine(
                        im_left, warp_coords_left, im_final_size
                    )
                    atlas_warped_right = cv2.warpAffine(
                        im_right, warp_coords_right, im_final_size
                    )
                    atlas_warped = cv2.bitwise_or(atlas_warped_left, atlas_warped_right)
                    ret, atlas_warped = cv2.threshold(
                        atlas_warped, 5, 255, cv2.THRESH_BINARY_INV
                    )
                    if not original_label:
                        atlas_label_left = cv2.warpAffine(
                            atlas_label_mask_left, warp_coords_left, im_final_size
                        )
                        atlas_label_right = cv2.warpAffine(
                            atlas_label_mask_right, warp_coords_right, im_final_size
                        )
                        atlas_label = cv2.bitwise_or(atlas_label_left, atlas_label_right)

                else:
                    pts_np = np.array(
                        [
                            dlc_pts[align_val][0],
                            dlc_pts[align_val][1],
                            dlc_pts[align_val][2],
                        ],
                        dtype=np.float32,
                    )
                    atlas_pts_np = np.array(
                        [
                            atlas_pts[align_val][0],
                            atlas_pts[align_val][1],
                            atlas_pts[align_val][2],
                        ],
                        dtype=np.float32,
                    )
                    warp_coords = cv2.getAffineTransform(pts_np, atlas_pts_np)
                    atlas_warped = cv2.warpAffine(im, warp_coords, (512, 512))
                    template = cv2.warpAffine(template, warp_coords, (512, 512))
                    # try:
                    #    atlas_warped = niftyreg_align(git_repo_base, atlas_warped, output_mask_path, n)
                    # except:
                    #    print("ERROR: could not use niftyreg to warp atlas {}! Please check inputs. Continuing...".format(str(n)))

            if atlas_to_brain_align:
                if len(atlas_pts_for_input[0]) == 2:
                    atlas_mask_left_warped = cv2.warpAffine(
                        atlas_mask_left, warp_coords, (512, 512)
                    )
                    atlas_mask_right_warped = cv2.warpAffine(
                        atlas_mask_right, warp_coords, (512, 512)
                    )
                    atlas_mask_warped = cv2.bitwise_or(
                        atlas_mask_left_warped, atlas_mask_right_warped
                    )
                if len(atlas_pts_for_input[0]) == 3:
                    atlas_mask_warped = cv2.warpAffine(atlas_mask, warp_coords, (512, 512))
                if len(atlas_pts_for_input[0]) >= 4:
                    atlas_mask_left_warped = cv2.warpAffine(
                        atlas_mask_left, warp_coords_left, (512, 512)
                    )
                    atlas_mask_right_warped = cv2.warpAffine(
                        atlas_mask_right, warp_coords_right, (512, 512)
                    )
                    atlas_mask_warped = cv2.bitwise_or(
                        atlas_mask_left_warped, atlas_mask_right_warped
                    )
                atlas_mask_warped = np.uint8(atlas_mask_warped)

            # Second alignment of brain atlas using cortical landmarks and piecewise affine transform
            print("Performing second transformation of atlas {}...".format(n))
            dst = atlas_warped

            # If a sensory map of the brain is provided, do a third alignment of the brain atlas using up to four peaks of
            # sensory activity
            if sensory_match:
                # COMMENT OUT FOR ALIGNING BRAIN TO ATLAS
                if atlas_to_brain_align:
                    dst = getMaskContour(
                        mask_dir,
                        atlas_warped,
                        sensory_peak_pts[align_val],
                        sensory_atlas_pts[align_val],
                        cwd,
                        align_val,
                        False,
                    )
                    atlas_mask_warped = getMaskContour(
                        mask_dir,
                        atlas_mask_warped,
                        sensory_peak_pts[align_val],
                        sensory_atlas_pts[align_val],
                        cwd,
                        align_val,
                        False,
                    )
                    atlas_mask_warped = cv2.resize(
                        atlas_mask_warped, (im.shape[0], im.shape[1])
                    )
                else:
                    dst = atlas_warped
        else:
            # If we're not using DeepLabCut...
            if atlas_to_brain_align and not use_voxelmorph:
                dst = cv2.bitwise_or(im_left, im_right)
                ret, dst = cv2.threshold(
                    dst, 5, 255, cv2.THRESH_BINARY_INV
                )
            else:
                dst = im
            dst = np.uint8(dst)

        voxelmorph_model_path = os.path.join(
            git_repo_base, "models", "voxelmorph", voxelmorph_model
        )

        atlas_first_transform_path = os.path.join(
            output_mask_path, "{}_atlas_first_transform.png".format(str(n))
        )

        if use_voxelmorph:
            if atlas_to_brain_align:
                _, flow = voxelmorph_align(
                    voxelmorph_model_path, br_vxm, template, exist_transform, flow_path
                )
            else:
                dst, flow = voxelmorph_align(
                    voxelmorph_model_path, dst, template, exist_transform, flow_path
                )
                # dst = dst * 255
                # ret, dst = cv2.threshold(
                #     dst, 5, 255, cv2.THRESH_BINARY_INV
                # )
            flow_path_after = os.path.join(output_mask_path, "{}_flow.npy".format(str(n)))
            np.save(flow_path_after, flow)
            if not exist_transform:
                if atlas_to_brain_align:
                    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                    dst = vxm_transform(dst_gray, flow_path_after)
                    ret, dst = cv2.threshold(
                        dst, 5, 255, cv2.THRESH_BINARY
                    )
                    dst = np.uint8(dst)
        io.imsave(atlas_first_transform_path, dst)

        if use_dlc:
            if atlas_to_brain_align:
                io.imsave(mask_warped_path, atlas_mask_warped)
            else:
                io.imsave(mask_warped_path, atlas_mask)
        else:
            io.imsave(mask_warped_path, dst)
            if atlas_to_brain_align:
                if use_voxelmorph:
                    atlas_mask_warped = atlas_mask
                else:
                    atlas_mask_warped = cv2.bitwise_or(
                        atlas_mask_left, atlas_mask_right
                    )
                atlas_mask_warped = cv2.cvtColor(atlas_mask_warped, cv2.COLOR_BGR2GRAY)
                atlas_mask_warped = vxm_transform(atlas_mask_warped, flow_path_after)
                ret, atlas_mask_warped = cv2.threshold(
                    atlas_mask_warped, 5, 255, cv2.THRESH_BINARY
                )
                atlas_mask_warped = np.uint8(atlas_mask_warped)
                original_label = True
            else:
                atlas_mask_warped = atlas_mask
                atlas_mask_warped = vxm_transform(atlas_mask_warped, flow_path_after)
            io.imsave(mask_warped_path, atlas_mask_warped)
        # Resize images back to 512x512
        dst = cv2.resize(dst, (im.shape[0], im.shape[1]))
        atlas_path = os.path.join(output_mask_path, "{}_atlas.png".format(str(n)))

        if atlas_to_brain_align:
            io.imsave(atlas_path, dst)
        else:
            brain_warped_path = os.path.join(
                output_mask_path, "{}_brain_warp.png".format(str(n))
            )
            io.imsave(brain_warped_path, dst)
            io.imsave(atlas_path, atlas)

        if atlas_to_brain_align:
            if original_label:
                atlas_label = []
            atlas_label = atlas_to_mask(
                atlas_path,
                mask_dir,
                mask_warped_path,
                output_mask_path,
                n,
                use_unet,
                atlas_to_brain_align,
                git_repo_base,
                olfactory_check,
                atlas_label,
                use_dlc
            )
            atlas_label_list.append(atlas_label)
        elif not use_dlc:
            io.imsave(os.path.join(output_mask_path, "{}.png".format(n)), dst)
        if bregma_present:
            bregma_val = int(bregma_index_list[n])
            bregma_list.append(dlc_pts[n][bregma_val])

    # Converts the transformed brain atlas into a segmentation method for the original brain image
    applyMask(
        brain_img_dir,
        output_mask_path,
        output_overlay_path,
        output_overlay_path,
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
        plot_landmarks,
        align_once,
        atlas_label_list,
        region_labels,
        original_label,
    )
