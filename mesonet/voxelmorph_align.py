"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
The method "vxm_data_generator" is adapted from VoxelMorph:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019). VoxelMorph: A Learning Framework for
    Deformable Medical Image Registration. IEEE Transactions on Medical Imaging, 38(8), 1788–1800.
    https://doi.org/10.1109/TMI.2019.2897538
VoxelMorph is distributed under the Apache License 2.0.
"""

from mesonet.mask_functions import *
from keras.models import *
import voxelmorph as vxm
from skimage.color import rgb2gray


def vxm_data_generator(x_data, batch_size=1):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    if batch_size == 1:
        x_data = rgb2gray(x_data)
        x_data = np.expand_dims(x_data, axis=0)
    vol_shape = x_data.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield inputs, outputs


def init_vxm_model(img_path, model_path):
    # configure unet features
    nb_features = [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

    # read_img = cv2.imread(img_path)

    # Since our input is a 2D image, we can take the shape from the first two dimensions in .shape
    inshape = img_path.shape[0:2]
    print(inshape)
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    lambda_param = 0.05
    loss_weights = [1, lambda_param]
    vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)
    vxm_model.load_weights(model_path)
    return vxm_model


def voxelmorph_align(model_path, img_path):
    vxm_model = init_vxm_model(img_path, model_path)
    val_generator = vxm_data_generator(img_path)
    val_input, _ = next(val_generator)

    # Makes predictions on each image
    results = vxm_model.predict(val_input)
    print(len(results))
    print(results[0].shape)
    # Saves output mask
    output_img = [img[0, :, :, 0] for img in results][0]
    print('Results saved!')
    return output_img
