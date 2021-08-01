"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='mesonet',
    version='1.0.4.1',
    author="Brandon Forys",
    author_email="brandon.forys@alumni.ubc.ca",
    description="An automatic brain region identification and segmentation toolbox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bf777/mesonet",
    install_requires=['imutils', 'scikit-image', 'scipy', 'numpy~=1.17.3', 'keras==2.3.1', 'opencv-python==4.4.0.46',
                      'Pillow', 'deeplabcut', 'pandas', 'matplotlib', 'python-polylabel',
                      'imgaug', 'voxelmorph', 'neurite'],
    packages=['mesonet', ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
