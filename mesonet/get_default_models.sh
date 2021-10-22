# MesoNet
# Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
# https://github.com/bf777/MesoNet
# Licensed under the Creative Commons Attribution 4.0 International License (see LICENSE for details)

# Get DeepLabCut model
osf -p svztu fetch 6_Landmark_estimation_model/atlas-DongshengXiao-2020-08-03.zip dlc/atlas-DongshengXiao-2020-08-03.zip

# Unzip DeepLabCut model
unzip -q dlc/atlas-DongshengXiao-2020-08-03.zip -d dlc

# Get U-Net model
osf -p svztu fetch 7_U-Net_model/DongshengXiao_brain_bundary.hdf5 models/DongshengXiao_brain_bundary.hdf5

# Get U-Net model for motif-based functional maps (MBFMs)
osf -p svztu fetch 7_U-Net_model/DongshengXiao_unet_motif_based_functional_atlas.hdf5 models/DongshengXiao_unet_motif_based_functional_atlas.hdf5

# Get VoxelMorph model
osf -p svztu fetch 8_VoxelMorph_model/VoxelMorph_Motif_based_functional_map_model_transformed1000.h5 models/voxelmorph/VoxelMorph_Motif_based_functional_map_model_transformed1000.h5