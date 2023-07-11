# MesoNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bf777/MesoNet/blob/master/mesonet_demo_colab.ipynb) [![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/1919930/tree) [![DOI](https://zenodo.org/badge/197092510.svg)](https://zenodo.org/badge/latestdoi/197092510) [![CC BY 4.0][cc-by-shield]][cc-by]

<img src="https://github.com/bf777/MesoNet/blob/master/img/logo.png" width="200" height="200" align="left">

**MesoNet** is a comprehensive Python toolbox for automated registration and segmentation of mesoscale mouse brain images.
You can use it to:
- Automatically identify cortical landmarks
- Register a brain atlas to your mesoscale calcium activity data (or vice versa)
- Segment brain data based on a brain atlas
- Use one of our machine learning  models (or train your own) to segment brain regions without using any landmarks!

**Read the MesoNet article in [Nature Communications](https://doi.org/10.1038/s41467-021-26255-2)**.

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
2. Activate the DeepLabCut environment (as described above, usually `activate DEEPLABCUT` on Windows or 
`source activate DEEPLABCUT` on Linux/Mac. 
3. Clone this git repository: `git clone https://github.com/bf777/MesoNet.git`
* NOTE: If you are on Windows, please clone the repository to a location on `C://` as the git repository search function does not currently support other drives.
5. Enter the git repository folder using `cd mesonet`, then run `pip install .` to install additional
dependencies for MesoNet (installation via pip coming soon!)

## Usage
Follow our tutorial [on the wiki](https://github.com/bf777/MesoNet/wiki/Quick-Start-Guide).

## Supported platforms
MesoNet has been tested on Windows 8.1, Windows 10, and Linux (Ubuntu 16.04 and Arch 5.7); it should also work on older versions of Windows and on MacOS, but these platforms have not been tested. It works with or without a GPU, but a GPU is _strongly_ recommended for faster training and processing. MesoNet can be used with or without a GUI, and can be run on headless platforms such as Google Colab.

## Contributors
[Dongsheng Xiao](https://github.com/DongshengXiao) designed the processing pipeline, collected data, trained the DeepLabCut, U-Net and VoxelMorph models provided with MesoNet, and developed the brain atlas alignment approach. [Brandon Forys](https://github.com/bf777) wrote the code of GUI and CLI. They are in the [Murphy Lab](https://murphylab.med.ubc.ca/) in UBC’s Department of Psychiatry.

## Citing
If you use our software, please cite the original research article:

Xiao, D., Forys, B.J., Vanni, M.P. et al. MesoNet allows automated scaling and segmentation of mouse mesoscale cortical maps using machine learning. Nat Commun 12, 5992 (2021). https://doi.org/10.1038/s41467-021-26255-2

BibTex:
```
@article{2021,
  doi = {10.1038/s41467-021-26255-2},
  url = {https://doi.org/10.1038/s41467-021-26255-2},
  year = {2021},
  month = oct,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {12},
  number = {1},
  author = {Dongsheng Xiao and Brandon J. Forys and Matthieu P. Vanni and Timothy H. Murphy},
  title = {{MesoNet} allows automated scaling and segmentation of mouse mesoscale cortical maps using machine learning}
}
```

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
