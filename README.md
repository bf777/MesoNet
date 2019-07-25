# MesoNet
MesoNet is a comprehensive Python toolbox for identifying, classifying, and exporting brain regions from mouse brain 
images.
You can use it to:
- Identify and classify brain regions based on an overlaid brain atlas with skull landmarks
- Identify and classify brain regions based on a functional image of the brain, with specific points of neural activation
- Use one of our machine learning  models (or train your own) to identify and classify brain regions without using any landmarks!

We offer an easy to use GUI, as well as a powerful command line interface (CLI) allowing you to integrate the toolbox
with your own neural imaging workflow.

MesoNet is built primarily on the U-net machine learning model
[(Ronneberger, Fischer, and Brox, 2015)](http://dx.doi.org/10.1007/978-3-319-24574-4_28),
as adapted in [zhixuhao](https://github.com/zhixuhao)'s [unet repository](https://github.com/zhixuhao/unet), as well as
[DeepLabCut](https://github.com/AlexEMG/DeepLabCut), [keras](https://github.com/keras-team/keras), and
[opencv](https://github.com/opencv/opencv).

## Installation
1. For DeepLabCut functionality (necessary for identifying brain atlas landmarks!),
[install and set up a DeepLabCut environment](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md).
We recommend using their supplied Anaconda environments.
2. Activate the DeepLabCut environment (as described above, usually `activate dlc-windowsGPU` or similar). 
3. Clone this git repository: `git clone https://github.com/bf777/MesoNet.git`
4. Enter the git repository folder using `cd mesonet`, then run `python setup.py install` to install additional
dependencies for MesoNet (installation via pip coming soon!)
5. Enter the main directory of MesoNet using `cd mesonet` again.

## Usage
Follow our tutorial [on the wiki](https://github.com/bf777/MesoNet/wiki/Quick-Start-Guide).

## Contributors
The code was written by [Brandon Forys](https://github.com/bf777) with contributions from Dongsheng Xiao and
Jeff LeDue in the [Murphy Lab](http://www.neuroscience.ubc.ca/faculty/murphy.html) in UBC's Department of Psychiatry.
