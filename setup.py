"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the Creative Commons Attribution 4.0 International License (see LICENSE for details)
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='mesonet',
    version='1.0.5.3',
    author="Brandon Forys",
    author_email="brandon.forys@alumni.ubc.ca",
    description="An automatic brain region identification and segmentation toolbox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bf777/mesonet",
    install_requires=['imutils', 'scikit-image', 'scipy', 'numpy', 'keras==2.3.1', 'opencv-python',
                      'Pillow', 'deeplabcut', 'pandas', 'matplotlib', 'python-polylabel',
                      'imgaug', 'voxelmorph', 'neurite', 'osfclient', 'h5py', 'tables'],
    packages=['mesonet', ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
