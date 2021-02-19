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
import voxelmorph as vxm
from skimage.color import rgb2gray


def vxm_data_generator(x_data, template, batch_size=1):
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
        print(template.shape)
        template = rgb2gray(template)
        x_data = np.expand_dims(x_data, axis=0)
        template = np.expand_dims(template, axis=0)
    vol_shape = x_data.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, template.shape[0], size=batch_size)
        fixed_images = template[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.

        # NOTE: we don't currently use these output images in our analyses;
        # the inputs are put directly into vxm_model.predict().
        outputs = [fixed_images, zero_phi]

        yield inputs, outputs


def init_vxm_model(img_path, model_path):
    # configure unet features
    nb_features = [
        [32, 32, 32, 32],  # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

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


def vxm_transform(x_data, flow_path):
    # If we already have a deformation field that we want to apply to all data,
    # use this deformation field instead of computing a new one.

    # preliminary sizing
    flow_data = np.load(flow_path)
    x_data = rgb2gray(x_data)
    x_data = np.expand_dims(x_data, axis=0)
    x_data = x_data[..., np.newaxis]

    vol_size = x_data.shape[1:-1]

    print(x_data.shape)
    print(flow_data.shape)

    results = vxm.networks.Transform(vol_size,
                                     interp_method='linear', nb_feats=x_data.shape[-1]).predict([x_data, flow_data])
    return results


def voxelmorph_align(model_path, img_path, template, exist_transform, flow_path):
    if not exist_transform:
        vxm_model = init_vxm_model(img_path, model_path)
        val_generator = vxm_data_generator(img_path, template)
        val_input, _ = next(val_generator)

        # Makes predictions on each image
        results = vxm_model.predict(val_input)
        # Saves output mask
        output_img = [img[0, :, :, 0] for img in results][0]
        # Saves flow image to flow
        flow_img = results[1]
    else:
        print('using existing transform')
        results = vxm_transform(img_path, flow_path)
        # Saves output mask
        print(results.shape)
        output_img = results[0, :, :, 0]
        # Saves flow image to flow
        flow_img = ''

    print('Results saved!')
    return output_img, flow_img
