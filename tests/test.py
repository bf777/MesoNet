import os
import shutil
import cv2
import mesonet
import time
import platform
import numpy as np


def test_mesonet_process():
    if not os.path.isdir('test_output'):
        os.mkdir('test_output')

    if os.path.isdir(os.path.join('test_output', 'dlc_output')):
        shutil.rmtree(os.path.join('test_output', 'dlc_output'))
    if os.path.isdir(os.path.join('test_output', 'output_mask')):
        shutil.rmtree(os.path.join('test_output', 'output_mask'))
    if os.path.isdir(os.path.join('test_output', 'output_overlay')):
        shutil.rmtree(os.path.join('test_output', 'output_overlay'))
    if os.path.isfile(os.path.join('test_output', 'mesonet_test_config.yaml')):
        os.remove(os.path.join('test_output', 'mesonet_test_config.yaml'))

    input_file = os.path.join(os.getcwd(), 'test_input')
    output_file = os.path.join(os.getcwd(), 'test_output')
    config_file = mesonet.config_project(input_file, output_file, 'test')

    # Config file tests
    config_dict = mesonet.parse_yaml(config_file)
    assert len(config_dict['landmark_arr']) == 9
    assert config_dict['num_images'] == 10

    # Procedure tests
    t0 = time.time()
    mesonet.predict_regions(config_file)
    t1 = time.time()
    predict_regions_time = t1 - t0

    t2 = time.time()
    mesonet.predict_dlc(config_file)
    t3 = time.time()
    predict_dlc_time = t3 - t2

    print('Time to run mesonet.predict_regions on 10 images: {} s'.format(np.round(predict_regions_time, 2)))
    print('Time to run mesonet.predict_dlc on 10 images: {} s'.format(np.round(predict_dlc_time, 2)))

    print('System info: ')
    print('Platform: {}'.format(platform.platform()))
    print('Processor: {}'.format(platform.processor()))

    import tensorflow as tf
    if tf.test.gpu_device_name():
        print('GPU: yes')
    else:
        print('GPU: no')
        print('If you were expecting the GPU to work, please check '
              'https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/installation.md for GPU installation '
              'troubleshooting tips.')


    assert os.path.isdir(os.path.join(output_file, 'dlc_output'))
    assert os.path.isdir(os.path.join(output_file, 'output_mask'))
    assert os.path.isdir(os.path.join(output_file, 'output_overlay'))

    assert os.path.isfile(os.path.join(output_file, 'output_mask', '0.png'))
    test_im = cv2.imread(os.path.join(output_file, 'output_mask', '0.png'))
    assert test_im.shape == (512, 512, 3)


if __name__ == "__main__":
    test_mesonet_process()
