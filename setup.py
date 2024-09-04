"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the Creative Commons Attribution 4.0 International License (see LICENSE for details)
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# 'tables==3.8.0', 'deeplabcut[tf]',
# pip install "notebook<7.0.0" "tensorflow-macos<2.13.0" "tensorflow-metal"
setuptools.setup(
    name='mesonet',
    version='1.20',
    author="Brandon Forys",
    author_email="brandon.forys@psych.ubc.ca",
    description="An automatic brain region identification and segmentation toolbox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bf777/mesonet",
    install_requires=['notebook<7.0.0',
                      'tensorflow-macos==2.12.0 ; platform_system=="Darwin"',
                      'tensorflow-metal ; platform_system=="Darwin"',
                      'deeplabcut[gui]',
                      'tf_keras',
                      'keras==2.13.1',
                      'numpy==1.23.5',
                      'tensorpack',
                      'tf_slim',
                      'imutils', 'scikit-image', 'scipy', 'opencv-python',
                      'Pillow', 'pandas', 'matplotlib',
                      'python-polylabel', 'imgaug', 'voxelmorph',
                      'osfclient', 'h5py', 'protobuf'],
    packages=['mesonet', ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
