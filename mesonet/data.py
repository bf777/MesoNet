"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the Creative Commons Attribution 4.0 International License (see LICENSE for details)
This file has been adapted from data.py in https://github.com/zhixuhao/unet
"""
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans


def adjustData(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = (
            np.reshape(
                new_mask,
                (
                    new_mask.shape[0],
                    new_mask.shape[1] * new_mask.shape[2],
                    new_mask.shape[3],
                ),
            )
            if flag_multi_class
            else np.reshape(
                new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2])
            )
        )
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.7] = 1
        mask[mask <= 0.7] = 0
    return img, mask


def trainGenerator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    aug_dict,
    image_color_mode="grayscale",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    flag_multi_class=False,
    num_class=2,
    save_to_dir=None,
    target_size=(512, 512),
    seed=1,
):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
    )
    train_generator = zip(image_generator, mask_generator)
    for img, mask in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def testGenerator(
    test_path,
    num_image=60,
    target_size=(512, 512),
    flag_multi_class=False,
    as_gray=True,
):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img
