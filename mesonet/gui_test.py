"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
import fnmatch
import glob
import os
from tkinter import *  # Python 3.x
from tkinter import filedialog

from PIL import Image, ImageTk
import imageio
import threading

from mesonet.dlc_predict import DLCPredict, DLCPredictBehavior
from mesonet.predict_regions import predictRegion
from mesonet.utils import (
    config_project,
    find_git_repo,
    natural_sort_key,
    convert_to_png,
    parse_yaml,
)


class Gui(object):
    """
    The main GUI interface for applying MesoNet to a new dataset.
    """

    def __init__(self, git_repo, config_file):
        # The main window of the app
        self.root = Tk()
        self.root.resizable(False, False)

        if config_file:
            self.cwd = os.getcwd()
            cfg = parse_yaml(config_file)
            self.atlas = cfg["atlas"]
            self.sensory_align = cfg["sensory_match"]
            self.sensoryName = cfg["sensory_path"]
            self.folderName = cfg["input_file"]
            self.saveFolderName = cfg["output"]
            self.mat_save = cfg["mat_save"]
            self.threshold = cfg["threshold"]
            self.git_repo_base = cfg["git_repo_base"]
            self.region_labels = cfg["region_labels"]
            self.landmark_arr = cfg["landmark_arr"]
            self.unet_select = cfg["use_unet"]
            self.dlc_select = cfg["use_dlc"]
            self.atlas_to_brain_align = cfg["atlas_to_brain_align"]
            self.model = cfg["model"]
            self.olfactory_check = cfg["olfactory_check"]
            self.plot_landmarks = cfg["plot_landmarks"]
            self.align_once = cfg["align_once"]
            self.original_label = cfg["original_label"]
            self.vxm_select = cfg["use_voxelmorph"]
            self.exist_transform = cfg["exist_transform"]
            self.vxm_model = cfg["voxelmorph_model"]
            self.templateName = cfg["template_path"]
            self.flowName = cfg["flow_path"]
        else:
            self.cwd = os.getcwd()
            self.folderName = self.cwd
            self.sensoryName = self.cwd
            self.saveFolderName = self.cwd
            self.threshold = 0.01  # 0.001
            self.vxm_model = "motif_model_atlas.h5"
            self.flowName = self.cwd
            self.landmark_arr = []

        self.BFolderName = self.cwd
        self.saveBFolderName = self.cwd

        self.j = -1
        self.delta = 0
        self.imgDisplayed = 0
        self.picLen = 0
        self.imageFileName = ""
        self.model = "unet_bundary.hdf5"
        self.status = 'Please select a folder with brain images at "Input Folder".'
        self.status_str = StringVar(self.root, value=self.status)
        self.haveMasks = False
        self.imgDisplayed = 0

        self.config_dir = "dlc"
        self.model_dir = "models"

        if git_repo == "" and not config_file:
            self.git_repo_base = find_git_repo()
        else:
            self.git_repo_base = os.path.join(git_repo, "mesonet")
        self.config_path = os.path.join(
            self.git_repo_base, self.config_dir, "config.yaml"
        )
        self.behavior_config_path = os.path.join(
            self.git_repo_base, self.config_dir, "behavior", " config.yaml"
        )
        self.model_top_dir = os.path.join(self.git_repo_base, self.model_dir)
        self.templateName = os.path.join(self.git_repo_base, "atlases", "templates")

        self.Title = self.root.title("MesoNet Analyzer")

        self.canvas = Canvas(self.root, width=512, height=512)
        self.canvas.grid(row=8, column=0, columnspan=4, rowspan=15, sticky=N + S + W)

        # Render model selector listbox
        self.modelSelect = []
        for file in os.listdir(self.model_top_dir):
            if fnmatch.fnmatch(file, "*.hdf5"):
                self.modelSelect.append(file)

        self.modelLabel = Label(
            self.root,
            text="If using U-net, select a model to analyze the brain regions:",
        )
        self.modelListBox = Listbox(self.root, exportselection=0)
        self.modelLabel.grid(row=0, column=4, columnspan=5, sticky=W + E + S)
        self.modelListBox.grid(
            row=1, rowspan=4, column=4, columnspan=5, sticky=W + E + N
        )
        for item in self.modelSelect:
            self.modelListBox.insert(END, item)

        if len(self.modelSelect) > 0:
            self.modelListBox.bind("<<ListboxSelect>>", self.onSelect)

        # Set file input and output
        self.fileEntryLabel = Label(self.root, text="Input folder")
        self.fileEntryLabel.grid(row=0, column=0, sticky=E + W)
        self.fileEntryButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenFile(0)
        )

        self.folderName_str = StringVar(self.root, value=self.folderName)
        self.fileEntryButton.grid(row=0, column=2, sticky=E)
        self.fileEntryBox = Entry(self.root, textvariable=self.folderName_str, width=50)
        self.fileEntryBox.grid(row=0, column=1, padx=5, pady=5)

        self.fileSaveLabel = Label(self.root, text="Save folder")
        self.fileSaveLabel.grid(row=1, column=0, sticky=E + W)
        self.fileSaveButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenFile(1)
        )

        self.saveFolderName_str = StringVar(self.root, value=self.saveFolderName)
        self.fileSaveButton.grid(row=1, column=2, sticky=E)
        self.fileSaveBox = Entry(
            self.root, textvariable=self.saveFolderName_str, width=50
        )
        self.fileSaveBox.grid(row=1, column=1, padx=5, pady=5)

        self.sensoryEntryLabel = Label(self.root, text="Sensory map folder")
        self.sensoryEntryLabel.grid(row=2, column=0, sticky=E + W)
        self.sensoryEntryButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenFile(2)
        )

        self.sensoryName_str = StringVar(self.root, value=self.sensoryName)
        self.sensoryEntryButton.grid(row=2, column=2, sticky=E)
        self.sensoryEntryBox = Entry(
            self.root, textvariable=self.sensoryName_str, width=50
        )
        self.sensoryEntryBox.grid(row=2, column=1, padx=5, pady=5)

        self.configDLCLabel = Label(self.root, text="DLC config folder")
        self.configDLCLabel.grid(row=3, column=0, sticky=E + W)
        self.configDLCButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenFile(3)
        )

        self.configDLCName_str = StringVar(self.root, value=self.config_path)
        self.configDLCButton.grid(row=3, column=2, sticky=E)
        self.configDLCEntryBox = Entry(
            self.root, textvariable=self.configDLCName_str, width=50
        )
        self.configDLCEntryBox.grid(row=3, column=1, padx=5, pady=5)

        self.gitLabel = Label(self.root, text="MesoNet git repo folder")
        self.gitLabel.grid(row=4, column=0, sticky=E + W)
        self.gitButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenFile(4)
        )

        self.git_str = StringVar(self.root, value=self.git_repo_base)
        self.gitButton.grid(row=4, column=2, sticky=E)
        self.gitEntryBox = Entry(self.root, textvariable=self.git_str, width=50)
        self.gitEntryBox.grid(row=4, column=1, padx=5, pady=5)

        # Set behavioural data files
        self.BfileEntryLabel = Label(self.root, text="Behavior input folder")
        self.BfileEntryLabel.grid(row=5, column=0, sticky=E + W)
        self.BfileEntryButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenBFile(0)
        )

        self.BfolderName_str = StringVar(self.root, value=self.folderName)
        self.BfileEntryButton.grid(row=5, column=2, sticky=E)
        self.BfileEntryBox = Entry(
            self.root, textvariable=self.BfolderName_str, width=50
        )
        self.BfileEntryBox.grid(row=5, column=1, padx=5, pady=5)

        self.BfileSaveLabel = Label(self.root, text="Behavior Save folder")
        self.BfileSaveLabel.grid(row=6, column=0, sticky=E + W)
        self.BfileSaveButton = Button(
            self.root, text="Browse...", command=lambda: self.OpenBFile(1)
        )

        self.saveBFolderName_str = StringVar(self.root, value=self.saveBFolderName)
        self.BfileSaveButton.grid(row=6, column=2, sticky=E)
        self.BfileSaveBox = Entry(
            self.root, textvariable=self.saveBFolderName_str, width=50
        )
        self.BfileSaveBox.grid(row=6, column=1, padx=5, pady=5)

        # Buttons for making predictions
        # Buttons below will only be active if a save file has been selected
        if not config_file:
            self.mat_save = BooleanVar()
            self.mat_save.set(True)
            self.atlas = BooleanVar()
            self.atlas.set(False)
            self.sensory_align = BooleanVar()
            self.sensory_align.set(False)
            self.region_labels = BooleanVar()
            self.region_labels.set(False)
            self.unet_select = BooleanVar()
            self.unet_select.set(True)
            self.dlc_select = BooleanVar()
            self.dlc_select.set(True)
            self.vxm_select = BooleanVar()
            self.vxm_select.set(True)
            self.olfactory_check = BooleanVar()
            self.olfactory_check.set(True)
            self.atlas_to_brain_align = BooleanVar()
            self.atlas_to_brain_align.set(True)
            self.plot_landmarks = BooleanVar()
            self.plot_landmarks.set(True)
            self.align_once = BooleanVar()
            self.align_once.set(False)
            self.original_label = BooleanVar()
            self.original_label.set(False)
            self.exist_transform = BooleanVar()
            self.exist_transform.set(False)

        self.landmark_left = BooleanVar()
        self.landmark_left.set(True)
        self.landmark_right = BooleanVar()
        self.landmark_right.set(True)
        self.landmark_bregma = BooleanVar()
        self.landmark_bregma.set(True)
        self.landmark_lambda = BooleanVar()
        self.landmark_lambda.set(True)

        self.landmark_top_left = BooleanVar()
        self.landmark_top_left.set(True)
        self.landmark_top_centre = BooleanVar()
        self.landmark_top_centre.set(True)
        self.landmark_top_right = BooleanVar()
        self.landmark_top_right.set(True)
        self.landmark_bottom_left = BooleanVar()
        self.landmark_bottom_left.set(True)
        self.landmark_bottom_right = BooleanVar()
        self.landmark_bottom_right.set(True)

        self.saveMatFileCheck = Checkbutton(
            self.root,
            text="Save predicted regions as .mat files",
            variable=self.mat_save,
            onvalue=True,
            offvalue=False,
        )
        self.saveMatFileCheck.grid(
            row=7, column=4, columnspan=5, padx=2, sticky=N + S + W
        )
        # self.regionLabelCheck = Checkbutton(self.root, text="Identify brain regions (experimental)",
        #                                     variable=self.region_labels)
        # self.regionLabelCheck.grid(row=8, column=4, padx=2, sticky=N + S + W)
        self.uNetCheck = Checkbutton(
            self.root,
            text="Use U-net for alignment",
            variable=self.unet_select,
            onvalue=True,
            offvalue=False,
        )
        self.uNetCheck.grid(
            row=8, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.dlcCheck = Checkbutton(
            self.root,
            text="Use DeepLabCut for alignment",
            variable=self.dlc_select,
            onvalue=True,
            offvalue=False,
        )
        self.dlcCheck.grid(
            row=9, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.vxmCheck = Checkbutton(
            self.root,
            text="Use VoxelMorph for alignment",
            variable=self.vxm_select,
            onvalue=True,
            offvalue=False,
        )
        self.vxmCheck.grid(
            row=10, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.olfactoryCheck = Checkbutton(
            self.root,
            text="Draw olfactory bulbs\n(uncheck if no olfactory bulb visible "
            "in all images)",
            variable=self.olfactory_check,
            onvalue=True,
            offvalue=False,
        )
        self.olfactoryCheck.grid(
            row=11, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.atlasToBrainCheck = Checkbutton(
            self.root,
            text="Align atlas to brain",
            variable=self.atlas_to_brain_align,
            onvalue=True,
            offvalue=False,
        )
        self.atlasToBrainCheck.grid(
            row=12, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.sensoryMapCheck = Checkbutton(
            self.root,
            text="Align using sensory map",
            variable=self.sensory_align,
            onvalue=True,
            offvalue=False,
        )
        self.sensoryMapCheck.grid(
            row=13, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.landmarkPlotCheck = Checkbutton(
            self.root,
            text="Plot DLC landmarks on final image",
            variable=self.plot_landmarks,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkPlotCheck.grid(
            row=14, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.alignOnceCheck = Checkbutton(
            self.root,
            text="Align based on first brain image only",
            variable=self.align_once,
            onvalue=True,
            offvalue=False,
        )
        self.alignOnceCheck.grid(
            row=15, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        self.origLabelCheck = Checkbutton(
            self.root,
            text="Use old label consistency method\n(less consistent)",
            variable=self.original_label,
            onvalue=True,
            offvalue=False,
        )
        self.origLabelCheck.grid(
            row=16, column=4, columnspan=5, padx=2, sticky=N + S + W
        )

        # Enable selection of landmarks for alignment
        self.landmarkLeftCheck = Checkbutton(
            self.root,
            text="Left",
            variable=self.landmark_left,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkLeftCheck.grid(row=17, column=4, padx=2, sticky=N + S + W)
        self.landmarkRightCheck = Checkbutton(
            self.root,
            text="Right",
            variable=self.landmark_right,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkRightCheck.grid(row=17, column=5, padx=2, sticky=N + S + W)
        self.landmarkBregmaCheck = Checkbutton(
            self.root,
            text="Bregma",
            variable=self.landmark_bregma,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkBregmaCheck.grid(row=17, column=6, padx=2, sticky=N + S + W)
        self.landmarkLambdaCheck = Checkbutton(
            self.root,
            text="Lambda",
            variable=self.landmark_lambda,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkLambdaCheck.grid(row=17, column=7, padx=2, sticky=N + S + W)

        self.landmarkTopLeftCheck = Checkbutton(
            self.root,
            text="Top left",
            variable=self.landmark_top_left,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkTopLeftCheck.grid(row=18, column=4, padx=2, sticky=N + S + W)
        self.landmarkTopCentreCheck = Checkbutton(
            self.root,
            text="Top centre",
            variable=self.landmark_top_centre,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkTopCentreCheck.grid(row=18, column=5, padx=2, sticky=N + S + W)
        self.landmarkTopRightCheck = Checkbutton(
            self.root,
            text="Top right",
            variable=self.landmark_top_right,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkTopRightCheck.grid(row=18, column=6, padx=2, sticky=N + S + W)
        self.landmarkBottomLeftCheck = Checkbutton(
            self.root,
            text="Bottom left",
            variable=self.landmark_bottom_left,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkBottomLeftCheck.grid(row=18, column=7, padx=2, sticky=N + S + W)
        self.landmarkBottomRightCheck = Checkbutton(
            self.root,
            text="Bottom right",
            variable=self.landmark_bottom_right,
            onvalue=True,
            offvalue=False,
        )
        self.landmarkBottomRightCheck.grid(row=18, column=8, padx=2, sticky=N + S + W)

        self.vxm_window_open = False
        if self.vxm_window_open:
            print("TEST")

        self.vxmSettingsButton = Button(
            self.root,
            text="Open VoxelMorph settings",
            command=lambda: self.open_voxelmorph_window(),
        )
        self.vxmSettingsButton.grid(
            row=19, column=4, columnspan=5, padx=2, sticky=N + S + W + E
        )

        # self.predictDLCButton = Button(
        #     self.root,
        #     text="Predict brain regions\nusing landmarks",
        #     command=lambda: self.PredictDLC(
        #         self.config_path,
        #         self.folderName,
        #         self.saveFolderName,
        #         False,
        #         int(self.sensory_align.get()),
        #         self.sensoryName,
        #         os.path.join(self.model_top_dir, self.model),
        #         self.picLen,
        #         int(self.mat_save.get()),
        #         self.threshold,
        #         True,
        #         self.haveMasks,
        #         self.git_repo_base,
        #         self.region_labels.get(),
        #         self.unet_select.get(),
        #         self.atlas_to_brain_align.get(),
        #         self.olfactory_check.get(),
        #         self.plot_landmarks.get(),
        #         self.align_once.get(),
        #         self.original_label.get(),
        #         self.vxm_select.get(),
        #         self.exist_transform.get(),
        #         os.path.join(self.model_top_dir, "voxelmorph", self.vxm_model),
        #         self.templateName,
        #         self.flowName,
        #     ),
        # )
        self.predictDLCButton = Button(
            self.root,
            text="Predict brain regions\nusing landmarks",
            command=lambda: self.EnterThread('predict_dlc')
        )
        self.predictDLCButton.grid(
            row=20, column=4, columnspan=5, padx=2, sticky=N + S + W + E
        )
        # self.predictAllImButton = Button(
        #     self.root,
        #     text="Predict brain regions directly\nusing pretrained U-net model",
        #     command=lambda: self.PredictRegions(
        #         self.folderName,
        #         self.picLen,
        #         self.model,
        #         self.saveFolderName,
        #         int(self.mat_save.get()),
        #         self.threshold,
        #         False,
        #         self.git_repo_base,
        #         self.region_labels.get(),
        #         self.olfactory_check.get(),
        #         self.unet_select.get(),
        #         self.plot_landmarks.get(),
        #         self.align_once.get(),
        #         self.region_labels.get(),
        #     ),
        # )
        self.predictAllImButton = Button(
            self.root,
            text="Predict brain regions directly\nusing pretrained U-net model",
            command=lambda: self.EnterThread('predict_regions')
        )
        self.predictAllImButton.grid(
            row=21, column=4, columnspan=5, padx=2, sticky=N + S + W + E
        )
        self.predictBehaviourButton = Button(
            self.root,
            text="Predict animal movements",
            command=lambda: DLCPredictBehavior(
                self.behavior_config_path, self.BFolderName, self.saveBFolderName
            ),
        )
        self.predictBehaviourButton.grid(
            row=22, column=4, columnspan=5, padx=2, sticky=N + S + W + E
        )

        # Image controls
        # Buttons below will only display if an image is displayed
        self.nextButton = Button(
            self.root,
            text="->",
            command=lambda: self.ImageDisplay(1, self.folderName, 0),
        )
        self.nextButton.grid(row=23, column=2, columnspan=2, sticky=E)
        self.previousButton = Button(
            self.root,
            text="<-",
            command=lambda: self.ImageDisplay(-1, self.folderName, 0),
        )
        self.previousButton.grid(row=23, column=0, columnspan=2, sticky=W)

        self.statusBar = Label(
            self.root, textvariable=self.status_str, bd=1, relief=SUNKEN, anchor=W
        )
        self.statusBar.grid(row=24, column=0, columnspan=9, sticky="we")

        # Bind right and left arrow keys to forward/backward controls
        self.root.bind("<Right>", self.forward)
        self.root.bind("<Left>", self.backward)

        # Buttons to run pre-defined pipelines
        # Label
        self.pipelinesLabel = Label(
            self.root,
            text="Quick Start: Automated pipelines"
        )

        self.pipelinesLabel.grid(
            row=0, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        # 1. Atlas-to-brain
        self.atlasToBrainButton = Button(
            self.root,
            text="1 - Atlas to brain",
            command=lambda: self.EnterThread('atlas_to_brain')
        )
        self.atlasToBrainButton.grid(
            row=1, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        # 2. Brain-to-atlas
        self.brainToAtlasButton = Button(
            self.root,
            text="2 - Brain to atlas",
            command=lambda: self.EnterThread('brain_to_atlas')
        )
        self.brainToAtlasButton.grid(
            row=2, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        # 3. Atlas-to-brain + sensory maps
        self.atlasToBrainSensoryButton = Button(
            self.root,
            text="3 - Atlas to brain +\nsensory maps",
            command=lambda: self.EnterThread('atlas_to_brain_sensory')
        )
        self.atlasToBrainSensoryButton.grid(
            row=3, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        # 4. Motif-based functional maps (MBFMs) + U-Net
        self.MBFMUNetButton = Button(
            self.root,
            text="4 - Motif-based functional maps (MBFMs) +\nMBFM-U-Net",
            command=lambda: self.EnterThread('MBFM_U_Net')
        )
        self.MBFMUNetButton.grid(
            row=4, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        # 5. Motif-based functional maps (MBFMs) + Brain-to-atlas + VoxelMorph
        self.MBFMBrainToAtlasVxmButton = Button(
            self.root,
            text="5 - Motif-based functional maps (MBFMs) +\nBrain-to-atlas + VoxelMorph",
            command=lambda: self.EnterThread('MBFM_brain_to_atlas_vxm')
        )
        self.MBFMBrainToAtlasVxmButton.grid(
            row=5, column=9, columnspan=1, padx=2, sticky=N + S + W + E
        )

        if self.saveFolderName == "" or self.imgDisplayed == 0:
            self.predictAllImButton.config(state="disabled")
            self.predictDLCButton.config(state="disabled")
            self.saveMatFileCheck.config(state="disabled")
            # self.regionLabelCheck.config(state='disabled')
            self.uNetCheck.config(state="disabled")
            self.dlcCheck.config(state="disabled")
            self.vxmCheck.config(state="disabled")
            self.olfactoryCheck.config(state="disabled")
            self.sensoryMapCheck.config(state="disabled")
            self.atlasToBrainCheck.config(state="disabled")
            self.predictBehaviourButton.config(state="disabled")
            self.landmarkPlotCheck.config(state="disabled")
            self.alignOnceCheck.config(state="disabled")
            self.origLabelCheck.config(state="disabled")

            self.landmarkLeftCheck.config(state="disabled")
            self.landmarkRightCheck.config(state="disabled")
            self.landmarkBregmaCheck.config(state="disabled")
            self.landmarkLambdaCheck.config(state="disabled")

            self.landmarkTopLeftCheck.config(state="disabled")
            self.landmarkTopCentreCheck.config(state="disabled")
            self.landmarkTopRightCheck.config(state="disabled")
            self.landmarkBottomLeftCheck.config(state="disabled")
            self.landmarkBottomRightCheck.config(state="disabled")

            self.atlasToBrainButton.config(state="disabled")
            self.brainToAtlasButton.config(state="disabled")
            self.atlasToBrainSensoryButton.config(state="disabled")
            self.MBFMUNetButton.config(state="disabled")
            self.MBFMBrainToAtlasVxmButton.config(state="disabled")

        if config_file:
            self.ImageDisplay(1, self.folderName, 1)
            self.predictAllImButton.config(state="normal")
            self.predictDLCButton.config(state="normal")
            self.saveMatFileCheck.config(state="normal")
            # self.regionLabelCheck.config(state='normal')
            self.uNetCheck.config(state="normal")
            self.dlcCheck.config(state="normal")
            self.vxmCheck.config(state="normal")
            self.olfactoryCheck.config(state="normal")
            self.atlasToBrainCheck.config(state="normal")
            self.sensoryMapCheck.config(state="normal")
            self.landmarkPlotCheck.config(state="normal")
            self.alignOnceCheck.config(state="normal")
            self.origLabelCheck.config(state="normal")

            self.landmarkLeftCheck.config(state="normal")
            self.landmarkRightCheck.config(state="normal")
            self.landmarkBregmaCheck.config(state="normal")
            self.landmarkLambdaCheck.config(state="normal")

            self.landmarkTopLeftCheck.config(state="normal")
            self.landmarkTopCentreCheck.config(state="normal")
            self.landmarkTopRightCheck.config(state="normal")
            self.landmarkBottomLeftCheck.config(state="normal")
            self.landmarkBottomRightCheck.config(state="normal")

    def OpenFile(self, openOrSave):
        if openOrSave == 0:
            newFolderName = filedialog.askdirectory(
                initialdir=self.cwd,
                title="Choose folder containing the brain images you want to "
                "analyze",
            )
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.folderName_str.set(newFolderName)
                self.folderName = newFolderName
                self.ImageDisplay(1, self.folderName, 1)
                self.statusHandler(
                    'Please select a folder to save your images to at "Save Folder".'
                )
            except:
                if self.folderName_str.get != newFolderName:
                    self.folderName_str.set(self.cwd)
                img_path_err = "No image file selected!"
                self.statusHandler(img_path_err)
        elif openOrSave == 1:
            newSaveFolderName = filedialog.askdirectory(
                initialdir=self.cwd, title="Choose folder for saving files"
            )
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.saveFolderName_str.set(newSaveFolderName)
                self.saveFolderName = newSaveFolderName
                self.predictAllImButton.config(state="normal")
                self.predictDLCButton.config(state="normal")
                self.saveMatFileCheck.config(state="normal")
                # self.regionLabelCheck.config(state='normal')
                self.uNetCheck.config(state="normal")
                self.dlcCheck.config(state="normal")
                self.vxmCheck.config(state="normal")
                self.olfactoryCheck.config(state="normal")
                self.atlasToBrainCheck.config(state="normal")
                self.sensoryMapCheck.config(state="normal")
                self.landmarkPlotCheck.config(state="normal")
                self.alignOnceCheck.config(state="normal")
                self.origLabelCheck.config(state="normal")

                self.landmarkLeftCheck.config(state="normal")
                self.landmarkRightCheck.config(state="normal")
                self.landmarkBregmaCheck.config(state="normal")
                self.landmarkLambdaCheck.config(state="normal")

                self.landmarkTopLeftCheck.config(state="normal")
                self.landmarkTopCentreCheck.config(state="normal")
                self.landmarkTopRightCheck.config(state="normal")
                self.landmarkBottomLeftCheck.config(state="normal")
                self.landmarkBottomRightCheck.config(state="normal")

                self.atlasToBrainButton.config(state="normal")
                self.brainToAtlasButton.config(state="normal")
                self.atlasToBrainSensoryButton.config(state="normal")
                self.MBFMUNetButton.config(state="normal")
                self.MBFMBrainToAtlasVxmButton.config(state="normal")

                self.statusHandler(
                    "Save folder selected! Choose an option on the right to begin your analysis."
                )
            except:
                if self.saveFolderName_str.get != newSaveFolderName:
                    self.saveFolderName_str.set(self.cwd)
                    save_path_err = "No save file selected!"
                print(save_path_err)
                self.statusHandler(save_path_err)
        elif openOrSave == 2:
            newSensoryName = filedialog.askdirectory(
                initialdir=self.cwd,
                title="Choose folder containing the sensory images you want to use",
            )
            try:
                self.sensoryName_str.set(newSensoryName)
                self.sensoryName = newSensoryName
                self.root.update()
            except:
                if self.sensoryName_str.get != newSensoryName:
                    self.sensoryName_str.set(self.cwd)
                sensory_path_err = "No sensory image file selected!"
                print(sensory_path_err)
                self.statusHandler(sensory_path_err)
        elif openOrSave == 3:
            newDLCName = filedialog.askopenfilename(
                initialdir=self.cwd,
                title="Choose folder containing the DLC config file",
            )
            try:
                self.configDLCName_str.set(newDLCName)
                self.config_path = newDLCName
                self.root.update()
            except:
                dlc_path_err = "No DLC config file selected!"
                self.statusHandler(dlc_path_err)
        elif openOrSave == 4:
            newGitName = filedialog.askdirectory(
                initialdir=self.cwd, title="Choose MesoNet git repository"
            )
            try:
                newGitName = os.path.join(newGitName, "mesonet")
                self.git_str.set(newGitName)
                self.git_repo_base = newGitName
                self.config_path = os.path.join(
                    self.git_repo_base, self.config_dir, "config.yaml"
                )
                self.configDLCName_str.set(self.config_path)
                self.behavior_config_path = os.path.join(
                    self.git_repo_base, self.config_dir, "behavior", " config.yaml"
                )
                self.model_top_dir = os.path.join(self.git_repo_base, self.model_dir)
                self.modelSelect = []
                for file in os.listdir(self.model_top_dir):
                    if fnmatch.fnmatch(file, "*.hdf5"):
                        self.modelSelect.append(file)
                self.modelListBox.delete(0, END)
                for item in self.modelSelect:
                    self.modelListBox.insert(END, item)
                self.root.update()
            except:
                dlc_path_err = "No git repo selected!"
                self.statusHandler(dlc_path_err)
        elif openOrSave == 5:
            newVxmTemplateName = filedialog.askdirectory(
                initialdir=self.cwd, title="Select VoxelMorph template directory"
            )
            try:
                self.templateName_str.set(newVxmTemplateName)
                self.templateName = newVxmTemplateName
            except:
                template_err = "No template folder selected!"
                self.statusHandler(template_err)
        elif openOrSave == 6:
            newVxmFlowName = filedialog.askopenfilename(
                initialdir=self.cwd, title="Select VoxelMorph flow file"
            )
            try:
                self.flowName_str.set(newVxmFlowName)
                self.flowName = newVxmFlowName
            except:
                template_err = "No template file selected!"
                self.statusHandler(template_err)

    def OpenBFile(self, openOrSave):
        if openOrSave == 0:
            newBFolderName = filedialog.askdirectory(
                initialdir=self.cwd,
                title="Choose folder containing the brain images you want to "
                "analyze",
            )
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.BfolderName_str.set(newBFolderName)
                self.BFolderName = newBFolderName
                self.root.update()
            except:
                print("No image file selected!")

        elif openOrSave == 1:
            newSaveBFolderName = filedialog.askdirectory(
                initialdir=self.cwd, title="Choose folder for saving files"
            )
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.saveBFolderName_str.set(newSaveBFolderName)
                self.saveBFolderName = newSaveBFolderName
                self.predictBehaviourButton.config(state="normal")
                self.statusHandler(
                    'Save folder selected! Click "Predict animal movements" to begin your analysis.'
                )
            except:
                save_path_err = "No save file selected!"
                print(save_path_err)
                self.statusHandler(save_path_err)

    def open_voxelmorph_window(self):
        # VoxelMorph options window
        self.vxm_window_open = True
        self.vxm_window = Toplevel(self.root)
        self.vxm_window.title("VoxelMorph settings")
        self.vxm_window.resizable(False, False)

        # Render voxelmorph model selector listbox
        self.vxm_model_select = []
        for file in os.listdir(os.path.join(self.model_top_dir, "voxelmorph")):
            if fnmatch.fnmatch(file, "*.h5"):
                self.vxm_model_select.append(file)

        self.vxmModelLabel = Label(
            self.vxm_window,
            text="If using VoxelMorph, select a model to align the brain image and atlas:",
        )
        self.vxmModelListBox = Listbox(self.vxm_window, exportselection=0)
        self.vxmModelLabel.grid(row=0, column=4, columnspan=5, sticky=W + E + S)
        self.vxmModelListBox.grid(
            row=1, rowspan=4, column=4, columnspan=5, sticky=W + E + N
        )
        for item in self.vxm_model_select:
            self.vxmModelListBox.insert(END, item)

        if len(self.vxm_model_select) > 0:
            self.vxmModelListBox.bind(
                "<<ListboxSelect>>", lambda event: self.onSelectVxm(event)
            )

        self.templateEntryLabel = Label(self.vxm_window, text="Template file location")
        self.templateEntryLabel.grid(row=0, column=0, sticky=E + W)

        self.templateName_str = StringVar(self.vxm_window, value=self.templateName)
        self.templateEntryButton = Button(
            self.vxm_window, text="Browse...", command=lambda: self.OpenFile(5)
        )

        self.templateEntryButton.grid(row=0, column=2, sticky=E)
        self.templateEntryBox = Entry(
            self.vxm_window, textvariable=self.templateName_str, width=50
        )
        self.templateEntryBox.grid(row=0, column=1, padx=5, pady=5)

        self.flowEntryLabel = Label(self.vxm_window, text="Flow file location")
        self.flowEntryLabel.grid(row=1, column=0, sticky=E + W)

        self.flowName_str = StringVar(self.vxm_window, value=self.flowName)
        self.flowEntryButton = Button(
            self.vxm_window, text="Browse...", command=lambda: self.OpenFile(6)
        )

        self.flowEntryButton.grid(row=1, column=2, sticky=E)
        self.flowEntryBox = Entry(
            self.vxm_window, textvariable=self.flowName_str, width=50
        )
        self.flowEntryBox.grid(row=1, column=1, padx=5, pady=5)

        self.existTransformCheck = Checkbutton(
            self.vxm_window,
            text="Use existing transformation",
            variable=self.exist_transform,
            onvalue=True,
            offvalue=False,
        )
        self.existTransformCheck.grid(
            row=2, column=0, columnspan=2, padx=2, sticky=N + S + W
        )

        self.vxm_window.mainloop()

    def ImageDisplay(self, delta, folderName, reset):
        # If input is .mat or .npy, convert to .png
        if glob.glob(os.path.join(folderName, "*.mat")) or glob.glob(
            os.path.join(folderName, "*.npy")
        ):
            convert_to_png(folderName)

        # Set up canvas on which images will be displayed
        is_tif = False
        self.imgDisplayed = 1
        self.root.update()
        if reset == 1:
            self.j = -1
        self.j += delta
        file_list = []
        tif_list = []
        if glob.glob(os.path.join(folderName, "*_mask_segmented.png")):
            file_list = glob.glob(os.path.join(folderName, "*_mask_segmented.png"))
            file_list.sort(key=natural_sort_key)
        elif glob.glob(os.path.join(folderName, "*_mask.png")):
            file_list = glob.glob(os.path.join(folderName, "*_mask.png"))
            file_list.sort(key=natural_sort_key)
        elif glob.glob(os.path.join(folderName, "*.png")):
            file_list = glob.glob(os.path.join(folderName, "*.png"))
            file_list.sort(key=natural_sort_key)
            print(file_list)
        elif glob.glob(os.path.join(folderName, "*.tif")):
            is_tif = True
            tif_list = glob.glob(os.path.join(folderName, "*.tif"))
            tif_stack = imageio.mimread(tif_list[0])
            file_list = tif_stack
        self.picLen = len(file_list)
        if self.j > self.picLen - 1:
            self.j = 0
        if self.j <= -1:
            self.j = self.picLen - 1
        if delta != 0:
            if is_tif:
                image_orig = Image.fromarray(file_list[0])
                self.imageFileName = tif_list[0]
                self.imageFileName = os.path.basename(self.imageFileName)
            else:
                self.imageFileName = os.path.basename(file_list[self.j])
                image = os.path.join(folderName, file_list[self.j])
                image_orig = Image.open(image)
            image_resize = image_orig.resize((512, 512))
            image_disp = ImageTk.PhotoImage(image_resize)
            self.canvas.create_image(256, 256, image=image_disp)
            label = Label(image=image_disp)
            label.image = image_disp
            self.root.update()
        imageName = StringVar(self.root, value=self.imageFileName)
        imageNum = "Image {}/{}".format(self.j + 1, self.picLen)
        imageNumPrep = StringVar(self.root, value=imageNum)
        imageNameLabel = Label(self.root, textvariable=imageName)
        imageNameLabel.grid(row=7, column=0, columnspan=2, sticky=W)
        imageNumLabel = Label(self.root, textvariable=imageNumPrep)
        imageNumLabel.grid(row=7, column=2, columnspan=2, sticky=E)

    def onSelect(self, event):
        w = event.widget
        selected = int(w.curselection()[0])
        new_model = self.modelListBox.get(selected)
        self.model = new_model
        print(self.model)
        self.root.update()

    def onSelectVxm(self, event):
        w_vxm = event.widget
        selected_vxm = int(w_vxm.curselection()[0])
        new_vxm_model = self.vxmModelListBox.get(selected_vxm)
        self.vxm_model = new_vxm_model
        print(self.vxm_model)
        self.root.update()

    def forward(self, event):
        self.ImageDisplay(1, self.folderName, 0)

    def backward(self, event):
        self.ImageDisplay(-1, self.folderName, 0)

    def statusHandler(self, status_str):
        self.status = status_str
        self.status_str.set(self.status)
        self.root.update()

    def chooseLandmarks(self):
        left = self.landmark_left.get()
        right = self.landmark_right.get()
        bregma = self.landmark_bregma.get()
        lambd = self.landmark_lambda.get()
        top_left = self.landmark_top_left.get()
        top_centre = self.landmark_top_centre.get()
        top_right = self.landmark_top_right.get()
        bottom_left = self.landmark_bottom_left.get()
        bottom_right = self.landmark_bottom_right.get()

        if left:
            self.landmark_arr.append(0)
        if top_left:
            self.landmark_arr.append(1)
        if bottom_left:
            self.landmark_arr.append(2)
        if top_centre:
            self.landmark_arr.append(3)
        if bregma:
            self.landmark_arr.append(4)
        if lambd:
            self.landmark_arr.append(5)
        if right:
            self.landmark_arr.append(6)
        if top_right:
            self.landmark_arr.append(7)
        if bottom_right:
            self.landmark_arr.append(8)

    def EnterThread(self, command):
        if command == 'predict_regions':
            threading.Thread(target=
            self.PredictRegions(
                self.folderName,
                self.picLen,
                self.model,
                self.saveFolderName,
                int(self.mat_save.get()),
                self.threshold,
                False,
                self.git_repo_base,
                self.region_labels.get(),
                self.olfactory_check.get(),
                self.unet_select.get(),
                self.plot_landmarks.get(),
                self.align_once.get(),
                self.region_labels.get(),
            )).start()
        elif command == 'predict_dlc':
            threading.Thread(target=
            self.PredictDLC(
                self.config_path,
                self.folderName,
                self.saveFolderName,
                False,
                int(self.sensory_align.get()),
                self.sensoryName,
                os.path.join(self.model_top_dir, 'DongshengXiao_brain_bundary.hdf5'),
                self.picLen,
                int(self.mat_save.get()),
                self.threshold,
                True,
                self.haveMasks,
                self.git_repo_base,
                self.region_labels.get(),
                self.unet_select.get(),
                self.dlc_select.get(),
                self.atlas_to_brain_align.get(),
                self.olfactory_check.get(),
                self.plot_landmarks.get(),
                self.align_once.get(),
                self.original_label.get(),
                self.vxm_select.get(),
                self.exist_transform.get(),
                os.path.join(self.model_top_dir, "voxelmorph", self.vxm_model),
                self.templateName,
                self.flowName
            )).start()
        elif command == 'atlas_to_brain':
            threading.Thread(target=
            self.PredictDLC(
                os.path.join(self.model_top_dir, 'atlas-DongshengXiao-2020-08-03', 'config.yaml'),
                self.folderName,
                self.saveFolderName,
                False,
                0,
                '',
                os.path.join(self.model_top_dir, 'DongshengXiao_brain_bundary.hdf5'),
                self.picLen,
                True,
                self.threshold,
                True,
                False,
                self.git_repo_base,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                '',
                '',
                ''
            )).start()
        elif command == 'brain_to_atlas':
            threading.Thread(target=
            self.PredictDLC(
                os.path.join(self.model_top_dir, 'atlas-DongshengXiao-2020-08-03', 'config.yaml'),
                self.folderName,
                self.saveFolderName,
                False,
                0,
                '',
                os.path.join(self.model_top_dir, 'DongshengXiao_brain_bundary.hdf5'),
                self.picLen,
                True,
                self.threshold,
                True,
                False,
                self.git_repo_base,
                False,
                True,
                True,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                '',
                '',
                ''
            )).start()
        elif command == 'atlas_to_brain_sensory':
            threading.Thread(target=
            self.PredictDLC(
                os.path.join(self.model_top_dir, 'atlas-DongshengXiao-2020-08-03', 'config.yaml'),
                self.folderName,
                self.saveFolderName,
                False,
                1,
                self.sensoryName,
                os.path.join(self.model_top_dir, 'DongshengXiao_brain_bundary.hdf5'),
                self.picLen,
                True,
                self.threshold,
                True,
                False,
                self.git_repo_base,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                '',
                '',
                ''
            )).start()
        elif command == 'MBFM_U_Net':
            threading.Thread(target=
            self.PredictRegions(
                self.folderName,
                self.picLen,
                os.path.join(self.model_top_dir, 'DongshengXiao_unet_motif_based_functional_atlas.hdf5'),
                self.saveFolderName,
                True,
                self.threshold,
                False,
                self.git_repo_base,
                False,
                True,
                True,
                False,
                False,
                False,
            )).start()
        elif command == 'MBFM_brain_to_atlas_vxm':
            threading.Thread(target=
            self.PredictDLC(
                os.path.join(self.model_top_dir, 'atlas-DongshengXiao-2020-08-03', 'config.yaml'),
                self.folderName,
                self.saveFolderName,
                False,
                0,
                '',
                os.path.join(self.model_top_dir, 'DongshengXiao_brain_bundary.hdf5'),
                self.picLen,
                True,
                self.threshold,
                True,
                False,
                self.git_repo_base,
                False,
                False,
                True,
                False,
                False,
                True,
                True,
                False,
                True,
                False,
                os.path.join(self.model_top_dir, "voxelmorph",
                             "VoxelMorph_Motif_based_functional_map_model_transformed1000.h5"),
                self.templateName,
                self.flowName
            )).start()

    def PredictRegions(
        self,
        input_file,
        num_images,
        model,
        output,
        mat_save,
        threshold,
        mask_generate,
        git_repo_base,
        region_labels,
        olfactory_check,
        use_unet,
        plot_landmarks,
        align_once,
        original_label,
    ):
        self.statusHandler("Processing...")
        atlas_to_brain_align = True
        pts = []
        pts2 = []
        atlas_label_list = []
        predictRegion(
            input_file,
            num_images,
            model,
            output,
            mat_save,
            threshold,
            mask_generate,
            git_repo_base,
            atlas_to_brain_align,
            pts,
            pts2,
            olfactory_check,
            use_unet,
            plot_landmarks,
            align_once,
            atlas_label_list,
            region_labels,
            original_label,
        )
        self.saveFolderName = output
        if mask_generate:
            self.folderName = os.path.join(self.saveFolderName, "output_mask")
            self.haveMasks = True
        else:
            self.folderName = os.path.join(self.saveFolderName, "output_overlay")
        self.statusHandler("Processing complete!")
        self.ImageDisplay(1, self.folderName, 1)

    def PredictDLC(
        self,
        config,
        input_file,
        output,
        atlas,
        sensory_match,
        sensory_path,
        model,
        num_images,
        mat_save,
        threshold,
        mask_generate,
        haveMasks,
        git_repo_base,
        region_labels,
        use_unet,
        use_dlc,
        atlas_to_brain_align,
        olfactory_check,
        plot_landmarks,
        align_once,
        original_label,
        use_voxelmorph,
        exist_transform,
        voxelmorph_model,
        template_path,
        flow_path,
    ):
        self.statusHandler("Processing...")
        self.chooseLandmarks()
        atlas_label_list = []
        coords_input_file = ''
        # if mask_generate and not haveMasks and atlas_to_brain_align and use_unet:
        if mask_generate and not haveMasks and use_unet:
            pts = []
            pts2 = []
            predictRegion(
                input_file,
                num_images,
                model,
                output,
                mat_save,
                threshold,
                mask_generate,
                git_repo_base,
                atlas_to_brain_align,
                pts,
                pts2,
                olfactory_check,
                use_unet,
                plot_landmarks,
                align_once,
                atlas_label_list,
                region_labels,
                original_label,
            )
        DLCPredict(
            config,
            input_file,
            output,
            atlas,
            sensory_match,
            sensory_path,
            mat_save,
            threshold,
            git_repo_base,
            region_labels,
            self.landmark_arr,
            use_unet,
            use_dlc,
            atlas_to_brain_align,
            model,
            olfactory_check,
            plot_landmarks,
            align_once,
            original_label,
            use_voxelmorph,
            exist_transform,
            voxelmorph_model,
            template_path,
            flow_path,
            coords_input_file,
        )
        saveFolderName = output
        if not atlas:
            self.folderName = os.path.join(saveFolderName, "output_overlay")
        elif atlas:
            self.folderName = os.path.join(saveFolderName, "dlc_output")
        config_project(
            input_file,
            saveFolderName,
            "test",
            config=config,
            atlas=atlas,
            sensory_match=sensory_match,
            mat_save=mat_save,
            threshold=threshold,
            model=model,
            region_labels=region_labels,
            use_unet=use_unet,
            use_dlc=use_dlc,
            atlas_to_brain_align=atlas_to_brain_align,
            olfactory_check=olfactory_check,
            plot_landmarks=plot_landmarks,
            align_once=align_once,
            atlas_label_list=atlas_label_list,
            original_label=original_label,
            use_voxelmorph=use_voxelmorph,
            exist_transform=exist_transform,
            voxelmorph_model=voxelmorph_model,
            template_path=template_path,
            flow_path=flow_path,
        )
        self.statusHandler("Processing complete!")
        self.ImageDisplay(1, self.folderName, 1)


def gui(git_find, config_file):
    Gui(git_find, config_file).root.mainloop()
