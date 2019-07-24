import yaml
import os


def config_project(input_dir, output_dir, mode, model_name='unet.hdf5'):
    """
    Generates a config file (mesonet_config.yaml)
    :param input_dir: The directory containing the input brain images
    :param output_dir: The directory containing the output files
    :param output_dir: If train, generates a config file for training; if test, generates a config file for applying
    the model.
    :param model_name: (optional) Set a new name for the unet model to be trained. Default is 'unet.hdf5'.tr
    """

    if mode == 'test':
        filename = "mesonet_test_config.yaml"
        data = dict(
            config='dlc/config.yaml',
            input_file=input_dir,
            output=output_dir,
            atlas=False,
            landmark_atlas_img='atlases/landmarks_atlas_512_512.png',
            sensory_atlas_img='atlases/sensorymap_atlas_512_512.png',
            sensory_match=True,
            mat_save=1,
            threshold=0.001,
            num_images=10,
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
    with open(config_file, 'r') as stream:
        try:
            d = yaml.safe_load(stream)
            return d
        except yaml.YAMLError as exc:
            print(exc)
