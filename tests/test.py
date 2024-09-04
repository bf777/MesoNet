"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""

import os
import mesonet

if __name__ == '__main__':
    # Define data input and output folders for each of five pipelines
    git_repo_base = mesonet.utils.find_git_repo()

    input_file = os.path.join(git_repo_base, '..', 'tests', 'sample_data', 'pipeline1_2')
    input_file_sensory_raw = os.path.join(git_repo_base, '..', 'tests', 'sample_data', 'pipeline3_sensory', 'sensory_raw')
    input_file_sensory_maps = os.path.join(git_repo_base, '..', 'tests', 'sample_data', 'pipeline3_sensory', 'sensory_maps')
    input_file_MBFM = os.path.join(git_repo_base, '..', 'tests', 'sample_data', 'pipeline4_MBFM-U-Net')
    input_file_voxelmorph = os.path.join(git_repo_base, '..', 'tests', 'sample_data', 'pipeline5_VoxelMorph')

    output_file_atlas_brain = os.path.join(git_repo_base, '..', 'tests', 'results', 'mesonet_output_atlas_brain')
    output_file_brain_atlas = os.path.join(git_repo_base, '..', 'tests', 'results', 'mesonet_output_brain_atlas')
    output_file_sensory = os.path.join(git_repo_base, '..', 'tests', 'results', 'mesonet_output_sensory')
    output_file_MBFM_U_Net = os.path.join(git_repo_base, '..', 'tests', 'results', 'mesonet_output_MBFM_U_Net')
    output_file_voxelmorph = os.path.join(git_repo_base, '..', 'tests', 'results', 'mesonet_output_voxelmorph')

    # Try to make output folders
    if not os.path.isdir(output_file_atlas_brain):
        try:
            os.makedirs(output_file_atlas_brain)
        except OSError:
            print("Creation of the directory %s failed" % output_file_atlas_brain)
        else:
            print("Successfully created the directory %s " % output_file_atlas_brain)
    else:
        print("Directory %s already exists!" % output_file_atlas_brain)

    if not os.path.isdir(output_file_brain_atlas):
        try:
            os.makedirs(output_file_brain_atlas)
        except OSError:
            print("Creation of the directory %s failed" % output_file_brain_atlas)
        else:
            print("Successfully created the directory %s " % output_file_brain_atlas)
    else:
        print("Directory %s already exists!" % output_file_brain_atlas)

    if not os.path.isdir(output_file_sensory):
        try:
            os.makedirs(output_file_sensory)
        except OSError:
            print("Creation of the directory %s failed" % output_file_sensory)
        else:
            print("Successfully created the directory %s " % output_file_sensory)
    else:
        print("Directory %s already exists!" % output_file_sensory)

    if not os.path.isdir(output_file_MBFM_U_Net):
        try:
            os.makedirs(output_file_MBFM_U_Net)
        except OSError:
            print("Creation of the directory %s failed" % output_file_MBFM_U_Net)
        else:
            print("Successfully created the directory %s " % output_file_MBFM_U_Net)
    else:
        print("Directory %s already exists!" % output_file_MBFM_U_Net)

    if not os.path.isdir(output_file_voxelmorph):
        try:
            os.makedirs(output_file_voxelmorph)
        except OSError:
            print("Creation of the directory %s failed" % output_file_voxelmorph)
        else:
            print("Successfully created the directory %s " % output_file_voxelmorph)
    else:
        print("Directory %s already exists!" % output_file_voxelmorph)

    # Define name of U-Net model
    model_name = os.path.join(git_repo_base, 'models', 'DongshengXiao_brain_bundary.hdf5')
    u_net_only_model_name = os.path.join(git_repo_base, 'models', 'DongshengXiao_unet_motif_based_functional_atlas.hdf5')

    # Define name of DeepLabCut model
    dlc_model_name = 'atlas-DongshengXiao-2020-08-03'

    # Define name of VoxelMorph model
    voxelmorph_model_name = 'VoxelMorph_Motif_based_functional_map_model_transformed1000.h5'

    model = model_name
    print(model)
    u_net_only_model = u_net_only_model_name
    dlc_config = os.path.join(git_repo_base, 'dlc', dlc_model_name, 'config.yaml')

    # Run without DLC GUI (headless)
    os.environ["DLClight"] = "True"

    # Prepare a MesoNet project for predicting pose locations using five default pipelines

    # 1. Atlas to brain
    # Atlas-to-brain warp with U-Net and DeepLabCut
    print('\n1. Atlas-to-brain warp with U-Net and DeepLabCut')
    config_file_atlas_brain = mesonet.config_project(input_file, output_file_atlas_brain, 'test',
                                                     atlas_to_brain_align=True, use_voxelmorph=False,
                                                     use_unet=True, use_dlc=True,
                                                     sensory_match=False, mat_save=False, config=dlc_config, model=model)

    # Identify outer edges of the cortex
    mesonet.predict_regions(config_file_atlas_brain)

    print('Output cortex masks can be found in `results/mesonet_output_atlas_brain/output_mask`')

    # Identify and use cortical landmarks to align the atlas to the brain
    mesonet.predict_dlc(config_file_atlas_brain)

    print('Output aligned brain images and atlases can be found in `results/mesonet_output_atlas_brain/output_overlay`')

    # 2. Brain to atlas
    # Brain-to-atlas warp with DeepLabCut
    print('\n2. Brain-to-atlas warp with DeepLabCut')
    config_file_brain_atlas = mesonet.config_project(input_file, output_file_brain_atlas, 'test',
                                                     atlas_to_brain_align=False, use_voxelmorph=False,
                                                     use_unet=True, use_dlc=True, sensory_match=False,
                                                     mat_save=False, olfactory_check=True, config=dlc_config, model=model)

    # Identify outer edges of the cortex
    mesonet.predict_regions(config_file_brain_atlas)

    print('Output cortex masks can be found in `results/mesonet_output_brain_atlas/output_mask`')

    # Identify and use cortical landmarks to align the brain to the atlas
    mesonet.predict_dlc(config_file_brain_atlas)

    print('Output aligned brain images and atlases can be found in `results/mesonet_output_brain_atlas/output_overlay`')

    # 3. Atlas to brain + sensory
    # Atlas-to-brain warp with U-Net, DeepLabCut, and sensory maps
    print('\n3. Atlas-to-brain warp with U-Net, DeepLabCut, and sensory maps')
    config_file_sensory = mesonet.config_project(input_file_sensory_raw, output_file_sensory, 'test',
                                                 atlas_to_brain_align=True, use_voxelmorph=False, use_unet=True,
                                                 use_dlc=True, sensory_match=True, sensory_path=input_file_sensory_maps,
                                                 mat_save=False, config=dlc_config, model=model)

    # Identify outer edges of the cortex
    mesonet.predict_regions(config_file_sensory)

    print('Output cortex masks can be found in `results/mesonet_output_sensory/output_mask`')

    # Identify and use cortical landmarks to align the atlas to the brain, itaking into account peaks of sensory activity
    # that are common across animals
    mesonet.predict_dlc(config_file_sensory)

    print('Output aligned brain images and atlases can be found in `results/mesonet_output_sensory/output_overlay`')

    # 4. MBFM + U-Net
    # Motif-based functional maps (MBFMs) with atlas directly applied using U-Net
    print('\n4. Motif-based functional maps (MBFMs) with atlas directly applied using U-Net')
    config_file_MBFM_U_Net = mesonet.config_project(input_file_MBFM, output_file_MBFM_U_Net, 'test',
                                                    atlas_to_brain_align=True, use_voxelmorph=False, use_unet=True,
                                                    use_dlc=False, sensory_match=False, mat_save=False, mask_generate=False,
                                                    config=dlc_config, model=u_net_only_model)

    # Identify outer edges of the cortex, then apply atlas directly using U-Net
    mesonet.predict_regions(config_file_MBFM_U_Net)

    print('Output cortex masks can be found in `results/mesonet_output_MBFM_U_Net/output_mask`')
    print('Output aligned brain images and atlases can be found in `results/mesonet_output_MBFM_U_Net/output_overlay`')

    # 5. VoxelMorph
    # Local deformation warp with VoxelMorph and DeepLabCut
    print('\n5. Local deformation warp with VoxelMorph and DeepLabCut')
    config_file_voxelmorph = mesonet.config_project(input_file_voxelmorph, output_file_voxelmorph, 'test',
                                                    atlas_to_brain_align=False, use_voxelmorph=True, use_unet=True,
                                                    use_dlc=True, sensory_match=False, mat_save=False, config=dlc_config,
                                                    model=model, align_once=True, olfactory_check=True,
                                                    voxelmorph_model=voxelmorph_model_name)

    # Identify outer edges of the cortex
    mesonet.predict_regions(config_file_voxelmorph)

    print('Output cortex masks can be found in `results/mesonet_output_voxelmorph/output_mask`')

    # Identify and use cortical landmarks, then use VoxelMorph to align further, for brain to atlas
    mesonet.predict_dlc(config_file_voxelmorph)

    print('Output aligned brain images and atlases can be found in `results/mesonet_output_voxelmorph/output_overlay`')
    print('Test complete. All default MesoNet pipelines are fully functional!')
