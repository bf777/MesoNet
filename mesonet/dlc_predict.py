"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
import deeplabcut
from mesonet.atlas_brain_matching import atlasBrainMatch
from mesonet.utils import parse_yaml
import cv2
import glob
import os


def DLCPredict(config, input_file, output, atlas, sensory_match, mat_save, threshold):
    """
    Takes a directory of brain images and predicts cortical landmark locations (left and right suture, bregma, and
    lambda) using a DeepLabCut model.
    :param config: The path to the DeepLabCut configuration file.
    :param input_file: The folder containing the brain images to be analyzed.
    :param output: The folder to which we save the output brain image, labelled with the predicted locations of each
    landmark.
    :param atlas: Checks if a brain atlas is to be aligned with the brain image using landmarks
    (based on choice made in GUI).
    """
    img_array = []
    if sensory_match == 1:
        sensory_match = True
    else:
        sensory_match = False
    if sensory_match:
        sensory_img_dir = os.path.join(input_file, 'sensory')
    else:
        sensory_img_dir = ''
    for filename in glob.glob(os.path.join(input_file, '*.png')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    if len(img_array) > 0:
        video_output_path = os.path.join(output, 'dlc_output')
        video_name = os.path.join(video_output_path, 'tmp_video.mp4')

        if not os.path.isdir(video_output_path):
            os.mkdir(video_output_path)
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 30, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        deeplabcut.analyze_videos(config, [video_output_path], videotype='.mp4', save_as_csv=True)
        deeplabcut.create_labeled_video(config, [video_name], filtered=True)
        for filename in glob.glob(os.path.join(video_output_path, 'tmp_videoDeepCut*.*')):
            try:
                if '.mp4' in filename:
                    output_video_name = filename
                elif '.csv' in filename:
                    coords_input = filename
            except FileNotFoundError:
                print(
                    "Please ensure that an output video and corresponding datafile from DeepLabCut are in the folder!")

        cap = cv2.VideoCapture(output_video_name)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(video_output_path, '{}.png'.format(str(i))), frame)
            i += 1

        cap.release()
        cv2.destroyAllWindows()

        if not atlas:
             atlasBrainMatch(input_file, sensory_img_dir, coords_input, sensory_match, mat_save, threshold)


def predict_dlc(config_file):
    """
    Loads parameters into DLCPredict from config file.
    :param config_file: The full path to a MesoNet config file (generated using mesonet.config_project())
    """
    cfg = parse_yaml(config_file)
    config = cfg['config']
    atlas = cfg['atlas']
    sensory_match = cfg['sensory_match']
    input_file = cfg['input_file']
    output = cfg['output']
    mat_save = cfg['mat_save']
    threshold = cfg['threshold']
    DLCPredict(config, input_file, output, atlas, sensory_match, mat_save, threshold)
