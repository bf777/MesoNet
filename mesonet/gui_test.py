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

from mesonet.dlc_predict import DLCPredict, DLCPredictBehavior
from mesonet.predict_regions import predictRegion
from mesonet.utils import config_project, find_git_repo


class Gui:
    """
    The main GUI interface for applying MesoNet to a new dataset.
    """
    def __init__(self):
        # The main window of the app
        self.root = Tk()
        self.root.resizable(False, False)

        self.cwd = os.getcwd()
        self.folderName = self.cwd
        self.sensoryName = self.cwd
        self.BFolderName = self.cwd
        self.saveFolderName = self.cwd
        self.saveBFolderName = self.cwd
        self.threshold = 0.01  # 0.001
        self.haveMasks = False

        self.j = -1
        self.delta = 0
        self.imgDisplayed = 0
        self.picLen = 0
        self.imageFileName = ''
        self.model = 'unet_bundary.hdf5'
        self.status = 'Please select a folder with brain images at "Input Folder".'
        self.status_str = StringVar(self.root, value=self.status)
        self.haveMasks = False
        self.imgDisplayed = 0
        self.landmark_arr = []

        self.config_dir = 'dlc'
        self.model_dir = 'models'

        self.git_repo_base = find_git_repo()
        self.config_path = os.path.join(self.git_repo_base, self.config_dir, 'config.yaml')
        self.behavior_config_path = os.path.join(self.git_repo_base, self.config_dir, 'behavior', ' config.yaml')
        self.model_top_dir = os.path.join(self.git_repo_base, self.model_dir)

        self.Title = self.root.title("MesoNet Analyzer")

        self.canvas = Canvas(self.root, width=512, height=512)
        self.canvas.grid(row=7, column=0, columnspan=4, rowspan=12, sticky=N + S + W)

        # Render model selector listbox
        self.modelSelect = []
        for file in os.listdir(self.model_top_dir):
            if fnmatch.fnmatch(file, "*.hdf5"):
                self.modelSelect.append(file)

        self.modelLabel = Label(self.root, text="If using U-net, select a model to analyze the brain regions:")
        self.modelListBox = Listbox(self.root)
        self.modelLabel.grid(row=0, column=4, columnspan=5, sticky=W + E + S)
        self.modelListBox.grid(row=1, rowspan=4, column=4, columnspan=5, sticky=W + E + N)
        for item in self.modelSelect:
            self.modelListBox.insert(END, item)

        if len(self.modelSelect) > 0:
            self.modelListBox.bind('<<ListboxSelect>>', self.onSelect)

        # Set file input and output
        self.fileEntryLabel = Label(self.root, text="Input folder")
        self.fileEntryLabel.grid(row=0, column=0, sticky=E + W)
        self.fileEntryButton = Button(self.root, text="Browse...", command=lambda: self.OpenFile(0))

        self.folderName_str = StringVar(self.root, value=self.folderName)
        self.fileEntryButton.grid(row=0, column=2, sticky=E)
        self.fileEntryBox = Entry(self.root, textvariable=self.folderName_str, width=50)
        self.fileEntryBox.grid(row=0, column=1, padx=5, pady=5)

        self.fileSaveLabel = Label(self.root, text="Save folder")
        self.fileSaveLabel.grid(row=1, column=0, sticky=E + W)
        self.fileSaveButton = Button(self.root, text="Browse...", command=lambda: self.OpenFile(1))

        self.saveFolderName_str = StringVar(self.root, value=self.saveFolderName)
        self.fileSaveButton.grid(row=1, column=2, sticky=E)
        self.fileSaveBox = Entry(self.root, textvariable=self.saveFolderName_str, width=50)
        self.fileSaveBox.grid(row=1, column=1, padx=5, pady=5)

        self.sensoryEntryLabel = Label(self.root, text="Sensory map folder")
        self.sensoryEntryLabel.grid(row=2, column=0, sticky=E + W)
        self.sensoryEntryButton = Button(self.root, text="Browse...", command=lambda: self.OpenFile(2))

        self.sensoryName_str = StringVar(self.root, value=self.sensoryName)
        self.sensoryEntryButton.grid(row=2, column=2, sticky=E)
        self.sensoryEntryBox = Entry(self.root, textvariable=self.sensoryName_str, width=50)
        self.sensoryEntryBox.grid(row=2, column=1, padx=5, pady=5)

        self.configDLCLabel = Label(self.root, text="DLC config folder")
        self.configDLCLabel.grid(row=3, column=0, sticky=E + W)
        self.configDLCButton = Button(self.root, text="Browse...", command=lambda: self.OpenFile(3))

        self.configDLCName_str = StringVar(self.root, value=self.config_path)
        self.configDLCButton.grid(row=3, column=2, sticky=E)
        self.configDLCEntryBox = Entry(self.root, textvariable=self.configDLCName_str, width=50)
        self.configDLCEntryBox.grid(row=3, column=1, padx=5, pady=5)

        # Set behavioural data files
        self.BfileEntryLabel = Label(self.root, text="Behavior input folder")
        self.BfileEntryLabel.grid(row=4, column=0, sticky=E + W)
        self.BfileEntryButton = Button(self.root, text="Browse...", command=lambda: self.OpenBFile(0))

        self.BfolderName_str = StringVar(self.root, value=self.folderName)
        self.BfileEntryButton.grid(row=4, column=2, sticky=E)
        self.BfileEntryBox = Entry(self.root, textvariable=self.BfolderName_str, width=50)
        self.BfileEntryBox.grid(row=4, column=1, padx=5, pady=5)

        self.BfileSaveLabel = Label(self.root, text="Behavior Save folder")
        self.BfileSaveLabel.grid(row=5, column=0, sticky=E + W)
        self.BfileSaveButton = Button(self.root, text="Browse...", command=lambda: self.OpenBFile(1))

        self.saveBFolderName_str = StringVar(self.root, value=self.saveBFolderName)
        self.BfileSaveButton.grid(row=5, column=2, sticky=E)
        self.BfileSaveBox = Entry(self.root, textvariable=self.saveBFolderName_str, width=50)
        self.BfileSaveBox.grid(row=5, column=1, padx=5, pady=5)

        # Buttons for making predictions
        # Buttons below will only be active if a save file has been selected
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
        self.olfactory_check = BooleanVar()
        self.olfactory_check.set(True)
        self.atlas_to_brain_align = BooleanVar()
        self.atlas_to_brain_align.set(True)
        self.plot_landmarks = BooleanVar()
        self.plot_landmarks.set(True)
        self.align_once = BooleanVar()
        self.align_once.set(False)

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

        self.saveMatFileCheck = Checkbutton(self.root, text="Save predicted regions as .mat files",
                                            variable=self.mat_save, onvalue=True, offvalue=False)
        self.saveMatFileCheck.grid(row=7, column=4, columnspan=5, padx=2, sticky=N + S + W)
        # self.regionLabelCheck = Checkbutton(self.root, text="Identify brain regions (experimental)",
        #                                     variable=self.region_labels)
        # self.regionLabelCheck.grid(row=8, column=4, padx=2, sticky=N + S + W)
        self.uNetCheck = Checkbutton(self.root, text="Use U-net for alignment", variable=self.unet_select,
                                     onvalue=True, offvalue=False)
        self.uNetCheck.grid(row=8, column=4, columnspan=5, padx=2, sticky=N + S + W)

        self.olfactoryCheck = Checkbutton(self.root, text="Draw olfactory bulbs (uncheck if no olfactory bulb visible "
                                                          "in all images)", variable=self.olfactory_check,
                                          onvalue=True, offvalue=False)
        self.olfactoryCheck.grid(row=9, column=4, columnspan=5, padx=2, sticky=N + S + W)

        self.atlasToBrainCheck = Checkbutton(self.root, text="Align atlas to brain", variable=self.atlas_to_brain_align,
                                             onvalue=True, offvalue=False)
        self.atlasToBrainCheck.grid(row=10, column=4, columnspan=5, padx=2, sticky=N + S + W)

        self.sensoryMapCheck = Checkbutton(self.root, text="Align using sensory map", variable=self.sensory_align,
                                           onvalue=True, offvalue=False)
        self.sensoryMapCheck.grid(row=11, column=4, columnspan=5, padx=2, sticky=N + S + W)

        self.landmarkPlotCheck = Checkbutton(self.root, text="Plot DLC landmarks on final image",
                                             variable=self.plot_landmarks, onvalue=True, offvalue=False)
        self.landmarkPlotCheck.grid(row=12, column=4, columnspan=5, padx=2, sticky=N + S + W)

        self.alignOnceCheck = Checkbutton(self.root, text="Align based on first brain image only",
                                          variable=self.align_once, onvalue=True, offvalue=False)
        self.alignOnceCheck.grid(row=13, column=4, columnspan=5, padx=2, sticky=N + S + W)

        # Enable selection of landmarks for alignment
        self.landmarkLeftCheck = Checkbutton(self.root, text="Left",
                                             variable=self.landmark_left, onvalue=True, offvalue=False)
        self.landmarkLeftCheck.grid(row=14, column=4, padx=2, sticky=N + S + W)
        self.landmarkRightCheck = Checkbutton(self.root, text="Right",
                                              variable=self.landmark_right, onvalue=True, offvalue=False)
        self.landmarkRightCheck.grid(row=14, column=5, padx=2, sticky=N + S + W)
        self.landmarkBregmaCheck = Checkbutton(self.root, text="Bregma",
                                               variable=self.landmark_bregma, onvalue=True, offvalue=False)
        self.landmarkBregmaCheck.grid(row=14, column=6, padx=2, sticky=N + S + W)
        self.landmarkLambdaCheck = Checkbutton(self.root, text="Lambda",
                                               variable=self.landmark_lambda, onvalue=True, offvalue=False)
        self.landmarkLambdaCheck.grid(row=14, column=7, padx=2, sticky=N + S + W)

        self.landmarkTopLeftCheck = Checkbutton(self.root, text="Top left",
                                                variable=self.landmark_top_left, onvalue=True, offvalue=False)
        self.landmarkTopLeftCheck.grid(row=15, column=4, padx=2, sticky=N + S + W)
        self.landmarkTopCentreCheck = Checkbutton(self.root, text="Top centre",
                                                  variable=self.landmark_top_centre, onvalue=True, offvalue=False)
        self.landmarkTopCentreCheck.grid(row=15, column=5, padx=2, sticky=N + S + W)
        self.landmarkTopRightCheck = Checkbutton(self.root, text="Top right",
                                                 variable=self.landmark_top_right, onvalue=True, offvalue=False)
        self.landmarkTopRightCheck.grid(row=15, column=6, padx=2, sticky=N + S + W)
        self.landmarkBottomLeftCheck = Checkbutton(self.root, text="Bottom left",
                                                   variable=self.landmark_bottom_left, onvalue=True, offvalue=False)
        self.landmarkBottomLeftCheck.grid(row=15, column=7, padx=2, sticky=N + S + W)
        self.landmarkBottomRightCheck = Checkbutton(self.root, text="Bottom right",
                                                    variable=self.landmark_bottom_right, onvalue=True, offvalue=False)
        self.landmarkBottomRightCheck.grid(row=15, column=8, padx=2, sticky=N + S + W)

        self.predictDLCButton = Button(self.root, text="Predict brain regions\nusing landmarks",
                                       command=lambda: self.PredictDLC(self.config_path, self.folderName,
                                                                       self.saveFolderName, False,
                                                                       int(self.sensory_align.get()),
                                                                       self.sensoryName,
                                                                       os.path.join(self.model_top_dir, self.model),
                                                                       self.picLen,
                                                                       int(self.mat_save.get()), self.threshold, True,
                                                                       self.haveMasks,
                                                                       self.git_repo_base,
                                                                       self.region_labels.get(),
                                                                       self.unet_select.get(),
                                                                       self.atlas_to_brain_align.get(),
                                                                       self.olfactory_check.get(),
                                                                       self.plot_landmarks.get(),
                                                                       self.align_once.get()))
        self.predictDLCButton.grid(row=16, column=4, columnspan=5, padx=2, sticky=N + S + W + E)
        self.predictAllImButton = Button(self.root, text="Predict brain regions directly\nusing pretrained U-net model",
                                         command=lambda: self.PredictRegions(self.folderName, self.picLen, self.model,
                                                                             self.saveFolderName,
                                                                             int(self.mat_save.get()), self.threshold,
                                                                             False, self.git_repo_base,
                                                                             self.region_labels.get()))
        self.predictAllImButton.grid(row=17, column=4, columnspan=5, padx=2, sticky=N + S + W + E)
        self.predictBehaviourButton = Button(self.root, text="Predict animal movements",
                                             command=lambda: DLCPredictBehavior(self.behavior_config_path,
                                                                                self.BFolderName,
                                                                                self.saveBFolderName))
        self.predictBehaviourButton.grid(row=18, column=4, columnspan=5, padx=2, sticky=N + S + W + E)

        # Image controls
        # Buttons below will only display if an image is displayed
        self.nextButton = Button(self.root, text="->", command=lambda: self.ImageDisplay(1, self.folderName, 0))
        self.nextButton.grid(row=19, column=2, columnspan=2, sticky=E)
        self.previousButton = Button(self.root, text="<-", command=lambda: self.ImageDisplay(-1, self.folderName, 0))
        self.previousButton.grid(row=19, column=0, columnspan=2, sticky=W)

        self.statusBar = Label(self.root, textvariable=self.status_str, bd=1, relief=SUNKEN, anchor=W)
        self.statusBar.grid(row=20, column=0, columnspan=9, sticky='we')

        # Bind right and left arrow keys to forward/backward controls
        self.root.bind('<Right>', self.forward)
        self.root.bind('<Left>', self.backward)

        if self.saveFolderName == '' or self.imgDisplayed == 0:
            self.predictAllImButton.config(state='disabled')
            self.predictDLCButton.config(state='disabled')
            self.saveMatFileCheck.config(state='disabled')
            # self.regionLabelCheck.config(state='disabled')
            self.uNetCheck.config(state='disabled')
            self.olfactoryCheck.config(state='disabled')
            self.sensoryMapCheck.config(state='disabled')
            self.atlasToBrainCheck.config(state='disabled')
            self.predictBehaviourButton.config(state='disabled')
            self.landmarkPlotCheck.config(state='disabled')
            self.alignOnceCheck.config(state='disabled')

            self.landmarkLeftCheck.config(state='disabled')
            self.landmarkRightCheck.config(state='disabled')
            self.landmarkBregmaCheck.config(state='disabled')
            self.landmarkLambdaCheck.config(state='disabled')

            self.landmarkTopLeftCheck.config(state='disabled')
            self.landmarkTopCentreCheck.config(state='disabled')
            self.landmarkTopRightCheck.config(state='disabled')
            self.landmarkBottomLeftCheck.config(state='disabled')
            self.landmarkBottomRightCheck.config(state='disabled')

    def OpenFile(self, openOrSave):
        if openOrSave == 0:
            newFolderName = filedialog.askdirectory(initialdir=self.cwd,
                                                    title="Choose folder containing the brain images you want to "
                                                          "analyze")
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.folderName_str.set(newFolderName)
                self.folderName = newFolderName
                self.ImageDisplay(1, self.folderName, 1)
                self.statusHandler('Please select a folder to save your images to at "Save Folder".')
            except:
                self.folderName_str.set(self.cwd)
                img_path_err = "No image file selected!"
                self.statusHandler(img_path_err)
        elif openOrSave == 1:
            newSaveFolderName = filedialog.askdirectory(initialdir=self.cwd,
                                                           title="Choose folder for saving files")
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.saveFolderName_str.set(newSaveFolderName)
                self.saveFolderName = newSaveFolderName
                self.predictAllImButton.config(state='normal')
                self.predictDLCButton.config(state='normal')
                self.saveMatFileCheck.config(state='normal')
                # self.regionLabelCheck.config(state='normal')
                self.uNetCheck.config(state='normal')
                self.olfactoryCheck.config(state='normal')
                self.atlasToBrainCheck.config(state='normal')
                self.sensoryMapCheck.config(state='normal')
                self.landmarkPlotCheck.config(state='normal')
                self.alignOnceCheck.config(state='normal')

                self.landmarkLeftCheck.config(state='normal')
                self.landmarkRightCheck.config(state='normal')
                self.landmarkBregmaCheck.config(state='normal')
                self.landmarkLambdaCheck.config(state='normal')

                self.landmarkTopLeftCheck.config(state='normal')
                self.landmarkTopCentreCheck.config(state='normal')
                self.landmarkTopRightCheck.config(state='normal')
                self.landmarkBottomLeftCheck.config(state='normal')
                self.landmarkBottomRightCheck.config(state='normal')

                self.statusHandler("Save folder selected! Choose an option on the right to begin your analysis.")
            except:
                self.saveFolderName_str.set(self.cwd)
                save_path_err = "No save file selected!"
                print(save_path_err)
                self.statusHandler(save_path_err)
        elif openOrSave == 2:
            newSensoryName = filedialog.askdirectory(initialdir=self.cwd,
                                                    title="Choose folder containing the sensory images you want to use")
            try:
                self.sensoryName_str.set(newSensoryName)
                self.sensoryName = newSensoryName
                self.root.update()
            except:
                self.sensoryName_str.set(self.cwd)
                sensory_path_err = "No sensory image file selected!"
                print(sensory_path_err)
                self.statusHandler(sensory_path_err)
        elif openOrSave == 3:
            newDLCName = filedialog.askopenfilename(initialdir=self.cwd,
                                                    title="Choose folder containing the DLC config file")
            try:
                self.configDLCName_str.set(newDLCName)
                self.config_path = newDLCName
                self.root.update()
            except:
                dlc_path_err = "No DLC config file selected!"
                self.statusHandler(dlc_path_err)

    def OpenBFile(self, openOrSave):
        if openOrSave == 0:
            newBFolderName = filedialog.askdirectory(initialdir=self.cwd,
                                                     title="Choose folder containing the brain images you want to "
                                                           "analyze")
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.BfolderName_str.set(newBFolderName)
                self.BFolderName = newBFolderName
                self.root.update()
            except:
                print("No image file selected!")

        elif openOrSave == 1:
            newSaveBFolderName = filedialog.askdirectory(initialdir=self.cwd,
                                                         title="Choose folder for saving files")
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.saveBFolderName_str.set(newSaveBFolderName)
                self.saveBFolderName = newSaveBFolderName
                self.predictBehaviourButton.config(state='normal')
                self.statusHandler('Save folder selected! Click "Predict animal movements" to begin your analysis.')
            except:
                save_path_err = 'No save file selected!'
                print(save_path_err)
                self.statusHandler(save_path_err)

    def ImageDisplay(self, delta, folderName, reset):
        # Set up canvas on which images will be displayed
        is_tif = False
        self.imgDisplayed = 1
        self.root.update()
        if reset == 1:
            self.j = -1
        self.j += delta
        if glob.glob(os.path.join(folderName, '*_mask_segmented.png')):
            fileList = glob.glob(os.path.join(folderName, '*_mask_segmented.png'))
        elif glob.glob(os.path.join(folderName, '*_mask.png')):
            fileList = glob.glob(os.path.join(folderName, '*_mask.png'))
        elif glob.glob(os.path.join(folderName, '*.png')):
            fileList = glob.glob(os.path.join(folderName, '*.png'))
        elif glob.glob(os.path.join(folderName, '*.tif')):
            is_tif = True
            tif_list = glob.glob(os.path.join(folderName, '*.tif'))
            tif_stack = imageio.mimread(tif_list[0])
            fileList = tif_stack
        self.picLen = len(fileList)
        if self.j > self.picLen - 1:
            self.j = 0
        if self.j <= -1:
            self.j = self.picLen - 1
        if delta != 0:
            for file in fileList:
                if is_tif or fnmatch.fnmatch(file, os.path.join(folderName, "{}_mask_segmented.png".format(self.j))) \
                        or fnmatch.fnmatch(file, os.path.join(folderName, "{}.png".format(self.j))) or \
                        fnmatch.fnmatch(file, os.path.join(folderName, "{}_mask.png".format(self.j))):
                    if is_tif:
                        image_orig = Image.fromarray(file)
                        self.imageFileName = tif_list[0]
                        self.imageFileName = os.path.basename(self.imageFileName)
                    else:
                        self.imageFileName = os.path.basename(file)
                        image = os.path.join(folderName, file)
                        image_orig = Image.open(image)
                    image_resize = image_orig.resize((512, 512))
                    image_disp = ImageTk.PhotoImage(image_resize)
                    self.canvas.create_image(256, 256, image=image_disp)
                    label = Label(image=image_disp)
                    label.image = image_disp
                    self.root.update()
        imageName = StringVar(self.root, value=self.imageFileName)
        imageNum = 'Image {}/{}'.format(self.j + 1, self.picLen)
        imageNumPrep = StringVar(self.root, value=imageNum)
        imageNameLabel = Label(self.root, textvariable=imageName)
        imageNameLabel.grid(row=6, column=0, columnspan=2,  sticky=W)
        imageNumLabel = Label(self.root, textvariable=imageNumPrep)
        imageNumLabel.grid(row=6, column=2, columnspan=2, sticky=E)

    def onSelect(self, event):
        w = event.widget
        selected = int(w.curselection()[0])
        new_model = self.modelListBox.get(selected)
        self.model = new_model
        print(self.model)
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

    def PredictRegions(self, input_file, num_images, model, output, mat_save, threshold, mask_generate, git_repo_base,
                       region_labels):
        self.statusHandler('Processing...')
        atlas_to_brain_align = True
        predictRegion(input_file, num_images, model, output, mat_save, threshold, mask_generate, git_repo_base,
                      atlas_to_brain_align, region_labels)
        self.saveFolderName = output
        if mask_generate:
            self.folderName = os.path.join(self.saveFolderName, "output_mask")
            self.haveMasks = True
        else:
            self.folderName = self.saveFolderName
        self.statusHandler('Processing complete!')
        self.ImageDisplay(1, self.folderName, 1)

    def PredictDLC(self, config, input_file, output, atlas, sensory_match, sensory_path, model, num_images, mat_save,
                   threshold, mask_generate, haveMasks, git_repo_base, region_labels, use_unet, atlas_to_brain_align,
                   olfactory_check, plot_landmarks, align_once):
        self.statusHandler('Processing...')
        self.chooseLandmarks()
        # if atlas_to_brain_align == 1:
        #     atlas_to_brain_align = True
        # else:
        #     atlas_to_brain_align = False
        # if plot_landmarks == 1:
        #     plot_landmarks = True
        # else:
        #     plot_landmarks = False
        # if olfactory_check == 1:
        #     olfactory_check = True
        # else:
        #     olfactory_check = False
        # if use_unet == 1:
        #     use_unet = True
        # else:
        #     use_unet = False
        # if align_once == 1:
        #     align_once = True
        # else:
        #     align_once = False
        if mask_generate and not haveMasks and atlas_to_brain_align and use_unet:
            pts = []
            pts2 = []
            predictRegion(input_file, num_images, model, output, mat_save, threshold, mask_generate, git_repo_base,
                          atlas_to_brain_align, pts, pts2, olfactory_check, use_unet, plot_landmarks, align_once,
                          region_labels)
        DLCPredict(config, input_file, output, atlas, sensory_match, sensory_path,
                   mat_save, threshold, git_repo_base, region_labels, self.landmark_arr, use_unet, atlas_to_brain_align,
                   model, olfactory_check, plot_landmarks, align_once)
        saveFolderName = output
        if not atlas:
            self.folderName = os.path.join(saveFolderName, "output_overlay")
        elif atlas:
            self.folderName = os.path.join(saveFolderName, "dlc_output")
        config_project(input_file, saveFolderName, 'test', config=config, atlas=atlas, sensory_match=sensory_match,
                       mat_save=mat_save, threshold=threshold, model=model, region_labels=region_labels,
                       use_unet=use_unet, atlas_to_brain_align=atlas_to_brain_align, olfactory_check=olfactory_check,
                       plot_landmarks=plot_landmarks, align_once=align_once)
        self.statusHandler('Processing complete!')
        self.ImageDisplay(1, self.folderName, 1)


def gui():
    Gui().root.mainloop()
