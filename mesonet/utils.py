"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
import yaml
import glob
import os


def config_project(input_dir, output_dir, mode, model_name='unet.hdf5'):
    """
    Generates a config file (mesonet_train_config.yaml or mesonet_test_config.yaml, depending on whether you are
    applying an existing model or training a new one).
    :param input_dir: The directory containing the input brain images
    :param output_dir: The directory containing the output files
    :param mode: If train, generates a config file for training; if test, generates a config file for applying
    the model.
    :param model_name: (optional) Set a new name for the unet model to be trained. Default is 'unet.hdf5'
    """
    if mode == 'test':
        filename = "mesonet_test_config.yaml"
        num_images = len(glob.glob(os.path.join(input_dir, '*.png')))
        data = dict(
            config='dlc/config.yaml',
            input_file=input_dir,
            output=output_dir,
            atlas=False,
            sensory_match=False,
            mat_save=True,
            threshold=0.0001,
            num_images=num_images,
            model='models/unet_bundary.hdf5'
        )
    elif mode == 'train':
        filename = "mesonet_train_config.yaml"
        data = dict(
            input_file=input_dir,
            model_name=model_name,
            log_folder=output_dir
        )

    with open(os.path.join(output_dir, filename), 'w') as outfile:
        yaml.dump(data, outfile)


def parse_yaml(config_file):
    """
    Parses the config file and returns a dictionary with its parameters.
    :param config_file: The full path to a MesoNet config file (generated using mesonet.config_project())
    """
    with open(config_file, 'r') as stream:
        try:
            d = yaml.safe_load(stream)
            return d
        except yaml.YAMLError as exc:
            print(exc)
