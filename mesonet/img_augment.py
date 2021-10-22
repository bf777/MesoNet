"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the Creative Commons Attribution 4.0 International License (see LICENSE for details)
"""
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import glob
import os
import cv2
import pandas as pd
import skimage.io as io


def img_augment_run(input_path, output_path, coords_input, data_gen_args):
    flatten = lambda l: [obj for sublist in l for obj in sublist]
    img_list = glob.glob(os.path.join(input_path, "*.png"))
    print(img_list)

    coords = pd.read_csv(coords_input, header=[0, 1, 2], index_col=[0])
    coords_aug = coords.copy()

    for img_num, img_name in enumerate(img_list):
        img = cv2.imread(img_name)
        coord_row_x = coords.iloc[img_num, 0::2]
        coord_row_y = coords.iloc[img_num, 1::2]
        coords_list = [
            Keypoint(float(x), float(y)) for [x, y] in zip(coord_row_x, coord_row_y)
        ]
        keypoints = KeypointsOnImage(coords_list, shape=img.shape)
        seq = iaa.Sequential(
            [
                iaa.Multiply(
                    (
                        0.5 - data_gen_args["brightness_range"],
                        0.5 + data_gen_args["brightness_range"],
                    )
                ),
                iaa.Affine(
                    rotate=(
                        -1 * data_gen_args["rotation_range"],
                        data_gen_args["rotation_range"],
                    ),
                    scale=(
                        1 - data_gen_args["zoom_range"],
                        1 + data_gen_args["zoom_range"],
                    ),
                    shear=(
                        -1 * data_gen_args["shear_range"],
                        data_gen_args["shear_range"],
                    ),
                    translate_percent={
                        "x": (
                            -1 * data_gen_args["width_shift_range"],
                            data_gen_args["width_shift_range"],
                        ),
                        "y": (
                            -1 * data_gen_args["height_shift_range"],
                            data_gen_args["height_shift_range"],
                        ),
                    },
                ),
            ]
        )

        # Augment keypoints and images
        image_aug, kps_aug = seq(image=img, keypoints=keypoints)
        coords_aug.iloc[img_num, :] = flatten(
            [[kp.x, kp.y] for kp in kps_aug.keypoints]
        )
        idx_name = coords_aug.index[img_num]
        idx_basename = os.path.basename(idx_name)
        coords_aug.rename(
            index={
                idx_name: idx_name.replace(
                    idx_basename, "{}_aug.png".format(idx_basename.split(".")[0])
                )
            },
            inplace=True,
        )
        print(
            idx_name.replace(
                idx_basename, "{}_aug.png".format(idx_basename.split(".")[0])
            )
        )

        io.imsave(os.path.join(output_path, os.path.basename(img_name)), img)
        io.imsave(
            os.path.join(
                output_path,
                "{}_aug.png".format(os.path.basename(img_name).split(".")[0]),
            ),
            image_aug,
        )

    # Adapted from DeepLabCut (to facilitate conversion to DLC-compatible format):
    # https://github.com/DeepLabCut/DeepLabCut/blob/master/deeplabcut/generate_training_dataset/labeling_toolbox.py
    coords_aug.sort_index(inplace=True)
    coords_aug = coords_aug.append(coords)
    coords_aug.to_csv(
        os.path.join(
            output_path, "{}.csv".format(os.path.basename(coords_input).split(".")[0])
        )
    )
    coords_aug.to_hdf(
        os.path.join(
            output_path, "{}.h5".format(os.path.basename(coords_input).split(".")[0])
        ),
        "df_with_missing",
        format="table",
        mode="w",
    )


def img_augment(
    input_path,
    output_path,
    coords_input,
    brightness_range=0.3,
    rotation_range=0.3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
):
    data_gen_args = dict(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        zoom_range=zoom_range,
        shear_range=0.05,
    )
    img_augment_run(input_path, output_path, coords_input, data_gen_args)
