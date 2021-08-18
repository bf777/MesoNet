# MesoNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bf777/MesoNet/blob/master/mesonet_demo_colab.ipynb) [![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/1919930/tree)

<img src="https://github.com/bf777/MesoNet/blob/master/img/logo.png" width="200" height="200" align="left">

**MesoNet** is a comprehensive Python toolbox for automated registration and segmentation of mesoscale mouse brain images.
You can use it to:
- Automatically identify cortical landmarks
- Register a brain atlas to your mesoscale calcium activity data (or vice versa)
- Segment brain data based on a brain atlas
- Use one of our machine learning  models (or train your own) to segment brain regions without using any landmarks!

We developed atlas-to-brain and brain-to-atlas approaches to make the software flexible, easy to use and robust.

We offer an easy to use GUI, as well as a powerful command line interface (CLI) allowing you to integrate the toolbox with your own neural imaging workflow.

We also extend our pipeline to make use of functional sensory maps and spontaneous cortical activity motifs.

We provided six end-to-end automated pipelines to allow users to quickly output results from input images.

MesoNet is built primarily on the U-net machine learning model
[(Ronneberger, Fischer, and Brox, 2015)](http://dx.doi.org/10.1007/978-3-319-24574-4_28),
as adapted in [zhixuhao](https://github.com/zhixuhao)'s [unet repository](https://github.com/zhixuhao/unet), as well as
[DeepLabCut](https://github.com/AlexEMG/DeepLabCut), [keras](https://github.com/keras-team/keras), and
[opencv](https://github.com/opencv/opencv).

## Installation
1. For DeepLabCut functionality (necessary for identifying brain atlas landmarks!),
[install and set up a DeepLabCut environment](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md).
We recommend using their supplied Anaconda environments.
2. Activate the DeepLabCut environment (as described above, usually `activate DLC-GPU` on Windows or 
`source activate DLC-GPU` on Linux/Mac - replace `DLC-GPU` with `DLC-CPU` if you installed the CPU version of DeepLabCut). 
3. Clone this git repository: `git clone https://github.com/bf777/MesoNet.git`
4. Enter the git repository folder using `cd mesonet`, then run `python setup.py install` to install additional
dependencies for MesoNet (installation via pip coming soon!)

## Usage
Follow our tutorial [on the wiki](https://github.com/bf777/MesoNet/wiki/Quick-Start-Guide).

## Supported platforms
MesoNet has been tested on Windows 8.1, Windows 10, and Linux (Ubuntu 16.04 and Arch 5.7); it should also work
on older versions of Windows and on MacOS, but these platforms have not been tested. It works with or without a GPU, but
a GPU is _strongly_ recommended for faster training and processing.

## Contributors
[Dongsheng Xiao](https://github.com/DongshengXiao) designed the processing pipeline, collected data,trained the DeeLabCut, U-Net and VoxelMorph models provided with MesoNet, and developed the brain atlas alignment approach. [Brandon Forys](https://github.com/bf777) wrote the code of GUI and CLI. They are in the [Murphy Lab](https://murphylab.med.ubc.ca/) in UBC’s Department of Psychiatry.
