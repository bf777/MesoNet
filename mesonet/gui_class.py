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

from mesonet.dlc_predict import DLCPredict, DLCPredictBehavior
from mesonet.predict_regions import predictRegion
from mesonet.utils import config_project, find_git_repo


class Gui:
    def __init__(self):
        # The main window of the app
        self.root = Tk()
        self.root.resizable(False, False)

        self.cwd = os.getcwd()
        self.folderName = self.cwd
        self.sensoryName = self.cwd
        self.BFolderName = self.cwd
        self.saveFolderName = ''
        self.saveBFolderName = ''
        self.threshold = 0.001
        self.haveMasks = False

        self.j = -1
        self.delta = 0
        self.imgDisplayed = 0
        self.picLen = 0
        self.imageFileName = ''
        self.model = ''
        self.status = 'Please select a folder with brain images at "Input Folder".'
        self.status_str = StringVar(self.root, value=self.status)
        self.haveMasks = False
        self.imgDisplayed = 0

        self.config_dir = 'dlc'
        self.model_dir = 'models'

        self.git_repo_base = find_git_repo()
        self.config_path = os.path.join(self.git_repo_base, self.config_dir, 'config.yaml')
        self.behavior_config_path = os.path.join(self.git_repo_base, self.config_dir, 'behavior', ' config.yaml')
        self.model_top_dir = os.path.join(self.git_repo_base, self.model_dir)

        self.Title = self.root.title("MesoNet Analyzer")

        self.canvas = Canvas(self.root, width=512, height=512)
        self.canvas.grid(row=6, column=0, columnspan=4, rowspan=8, sticky=N + S + W)

        # Render model selector listbox
        self.FindModelInFolder()

        # Set file input and output
        self.fileEntryLabel = Label(self.root, text="Input folder")
        self.fileEntryLabel.grid(row=0, column=0, sticky=E + W)
        self.fileEntryButton = Button(self.root, text="Browse...", command=lambda: self.OpenFile(0))

        self.folderName_str = StringVar(self.root, value=self.folderName)
        self.fileEntryButton.grid(row=0, column=2, sticky=E)
        self.fileEntryBox = Entry(self.root, textvariable=self.folderName_str, width=60)
        self.fileEntryBox.grid(row=0, column=1, padx=5, pady=5)

        self.fileSaveLabel = Label(self.root, text="Save folder")
        self.fileSaveLabel.grid(row=1, column=0, sticky=E + W)
        self.fileSaveButton = Button(self.root, text="Browse...", command=lambda: self.OpenFile(1))

        self.saveFolderName_str = StringVar(self.root, value=self.saveFolderName)
        self.fileSaveButton.grid(row=1, column=2, sticky=E)
        self.fileSaveBox = Entry(self.root, textvariable=self.saveFolderName_str, width=60)
        self.fileSaveBox.grid(row=1, column=1, padx=5, pady=5)

        self.sensoryEntryLabel = Label(self.root, text="Sensory map folder")
        self.sensoryEntryLabel.grid(row=2, column=0, sticky=E + W)
        self.sensoryEntryButton = Button(self.root, text="Browse...", command=lambda: self.OpenFile(2))

        self.sensoryName_str = StringVar(self.root, value=self.sensoryName)
        self.sensoryEntryButton.grid(row=2, column=2, sticky=E)
        self.sensoryEntryBox = Entry(self.root, textvariable=self.sensoryName_str, width=60)
        self.sensoryEntryBox.grid(row=2, column=1, padx=5, pady=5)

        # Set behavioural data files
        self.BfileEntryLabel = Label(self.root, text="Behavior input folder")
        self.BfileEntryLabel.grid(row=3, column=0, sticky=E + W)
        self.BfileEntryButton = Button(self.root, text="Browse...", command=lambda: self.OpenBFile(0))

        self.BfolderName_str = StringVar(self.root, value=self.folderName)
        self.BfileEntryButton.grid(row=3, column=2, sticky=E)
        self.BfileEntryBox = Entry(self.root, textvariable=self.BfolderName_str, width=60)
        self.BfileEntryBox.grid(row=3, column=1, padx=5, pady=5)

        self.BfileSaveLabel = Label(self.root, text="Behavior Save folder")
        self.BfileSaveLabel.grid(row=4, column=0, sticky=E + W)
        self.BfileSaveButton = Button(self.root, text="Browse...", command=lambda: self.OpenBFile(1))

        self.saveBFolderName_str = StringVar(self.root, value=self.saveBFolderName)
        self.BfileSaveButton.grid(row=4, column=2, sticky=E)
        self.BfileSaveBox = Entry(self.root, textvariable=self.saveBFolderName_str, width=60)
        self.BfileSaveBox.grid(row=4, column=1, padx=5, pady=5)

        # Image controls
        # Buttons below will only display if an image is displayed
        self.nextButton = Button(self.root, text="->", command=lambda: self.ImageDisplay(1, self.folderName, 0))
        self.nextButton.grid(row=14, column=2, columnspan=1)
        self.previousButton = Button(self.root, text="<-", command=lambda: self.ImageDisplay(-1, self.folderName, 0))
        self.previousButton.grid(row=14, column=0, columnspan=1)

        self.statusBar = Label(self.root, textvariable=self.status_str, bd=1, relief=SUNKEN, anchor=W)
        self.statusBar.grid(row=15, column=0, columnspan=5, sticky='we')

        # Bind right and left arrow keys to forward/backward controls
        self.root.bind('<Right>', self.forward)
        self.root.bind('<Left>', self.backward)

        # Buttons for making predictions
        # Buttons below will only be active if a save file has been selected
        self.mat_save = IntVar()
        self.atlas = IntVar()
        self.sensory_align = IntVar()
        self.region_labels = IntVar()
        self.saveMatFileCheck = Checkbutton(self.root, text="Save predicted regions\nas .mat files",
                                            variable=self.mat_save)
        self.saveMatFileCheck.grid(row=7, column=4, padx=2, sticky=N + S + W)
        # self.regionLabelCheck = Checkbutton(self.root, text="Identify brain regions\n(experimental)",
        #                                     variable=self.region_labels)
        # self.regionLabelCheck.grid(row=8, column=4, padx=2, sticky=N + S + W)
        self.sensoryMapCheck = Checkbutton(self.root, text="Align using sensory map",
                                           variable=self.sensory_align)
        self.sensoryMapCheck.grid(row=9, column=4, padx=2, sticky=N + S + W)

        self.predictDLCButton = Button(self.root, text="Predict brain regions\nusing landmarks",
                                       command=lambda: self.PredictDLC(self.config_path, self.folderName,
                                                                       self.saveFolderName, False,
                                                                       int(self.sensory_align.get()),
                                                                       self.sensoryName,
                                                                       os.path.join(self.model_top_dir,
                                                                                    'unet_bundary.hdf5'),
                                                                       self.picLen,
                                                                       int(self.mat_save.get()), self.threshold, True,
                                                                       self.haveMasks,
                                                                       self.git_repo_base,
                                                                       self.region_labels.get()))
        self.predictDLCButton.grid(row=10, column=4, padx=2, sticky=N + S + W + E)
        self.predictAllImButton = Button(self.root, text="Predict brain regions directly\nusing pretrained U-net model",
                                         command=lambda: self.PredictRegions(self.folderName, self.picLen, self.model,
                                                                             self.saveFolderName,
                                                                             int(self.mat_save.get()), self.threshold,
                                                                             False, self.git_repo_base, self.region_labels.get()))
        self.predictAllImButton.grid(row=11, column=4, padx=2, sticky=N + S + W + E)
        self.predictBehaviourButton = Button(self.root, text="Predict animal movements",
                                             command=lambda: DLCPredictBehavior(self.behavior_config_path,
                                                                                self.BFolderName,
                                                                                self.saveBFolderName))
        self.predictBehaviourButton.grid(row=12, column=4, padx=2, sticky=N + S + W + E)

        if self.saveFolderName == '' or self.imgDisplayed == 0:
            self.predictAllImButton.config(state='disabled')
            self.predictDLCButton.config(state='disabled')
            self.saveMatFileCheck.config(state='disabled')
            # self.regionLabelCheck.config(state='disabled')
            self.sensoryMapCheck.config(state='disabled')
            self.predictBehaviourButton.config(state='disabled')

    def OpenFile(self, openOrSave):
        if openOrSave == 0:
            newFolderName = filedialog.askdirectory(initialdir=self.cwd,
                                                    title="Choose folder containing the brain images you want to analyze")
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.folderName_str.set(newFolderName)
                self.folderName = newFolderName
                self.ImageDisplay(1, self.folderName, 1)
                self.status = 'Please select a folder to save your images to at "Save Folder".'
                self.status_str.set(self.status)
                self.root.update()
            except:
                print("No image file selected!")
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
                self.sensoryMapCheck.config(state='normal')
                self.status = "Save folder selected! Choose an option on the right to begin your analysis."
                self.status_str.set(self.status)
                self.root.update()
            except:
                print("No save file selected!")
                self.status = "No save file selected!"
                self.status_str.set(self.status)
                self.root.update()
        elif openOrSave == 2:
            newSensoryName = filedialog.askdirectory(initialdir=self.cwd,
                                                    title="Choose folder containing the sensory images you want to use")
            try:
                self.sensoryName_str.set(newSensoryName)
                self.sensoryName = newSensoryName
                self.root.update()
            except:
                print("No sensory image file selected!")

    def OpenBFile(self, openOrSave):
        if openOrSave == 0:
            newBFolderName = filedialog.askdirectory(initialdir=self.cwd,
                                                     title="Choose folder containing the brain images you want to analyze")
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
                self.status = 'Save folder selected! Click "Predict animal movements" to begin your analysis.'
                self.status_str.set(self.status)
                self.root.update()
            except:
                print("No save file selected!")
                self.status = "No save file selected!"
                self.status_str.set(self.status)
                self.root.update()

    def ImageDisplay(self, delta, folderName, reset):
        # Set up canvas on which images will be displayed
        self.imgDisplayed = 1
        self.root.update()
        if reset == 1:
            self.j = -1
        self.j += delta
        if glob.glob(os.path.join(folderName, '*_mask_segmented.png')):
            fileList = glob.glob(os.path.join(folderName, '*_mask_segmented.png'))
        elif glob.glob(os.path.join(folderName, '*_mask.png')):
            fileList = glob.glob(os.path.join(folderName, '*_mask.png'))
        else:
            fileList = glob.glob(os.path.join(folderName, '*.png'))
        self.picLen = len(fileList)
        if self.j > self.picLen - 1:
            self.j = 0
        if self.j <= -1:
            self.j = self.picLen - 1
        if delta != 0:
            for file in fileList:
                if fnmatch.fnmatch(file, os.path.join(folderName, "{}_mask_segmented.png".format(self.j))) or \
                        fnmatch.fnmatch(file, os.path.join(folderName, "{}.png".format(self.j))) or \
                        fnmatch.fnmatch(file, os.path.join(folderName, "{}_mask.png".format(self.j))):
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
        imageNameLabel.grid(row=5, column=0, columnspan=1)
        imageNumLabel = Label(self.root, textvariable=imageNumPrep)
        imageNumLabel.grid(row=5, column=2, columnspan=1)

    def FindModelInFolder(self):
        modelSelect = []
        for file in os.listdir(self.model_top_dir):
            if fnmatch.fnmatch(file, "*.hdf5"):
                modelSelect.append(file)

        modelLabel = Label(self.root, text="If using U-net, select a model\nto analyze the brain regions:")
        modelListBox = Listbox(self.root)
        modelLabel.grid(row=5, column=4, sticky=S)
        modelListBox.grid(row=6, column=4, sticky=N)
        for item in modelSelect:
            modelListBox.insert(END, item)
        if len(modelSelect) > 0:
            def onselect(evt):
                w = evt.widget
                selected = int(w.curselection()[0])
                self.model = modelListBox.get(selected)
                self.model = os.path.join(self.git_repo_base, self.model_dir, self.model)
                self.root.update()

            modelListBox.bind('<<ListboxSelect>>', onselect)

    def forward(self, event):
        self.ImageDisplay(1, self.folderName, 0)

    def backward(self, event):
        self.ImageDisplay(-1, self.folderName, 0)

    def PredictRegions(self, input_file, num_images, model, output, mat_save, threshold, mask_generate, git_repo_base,
                       region_labels):
        self.status = "Processing..."
        self.status_str.set(self.status)
        self.root.update()
        predictRegion(input_file, num_images, model, output, mat_save, threshold, mask_generate, git_repo_base,
                      region_labels)
        self.saveFolderName = output
        if mask_generate:
            self.folderName = os.path.join(self.saveFolderName, "output_mask")
            self.haveMasks = True
        else:
            self.folderName = self.saveFolderName
        self.status = "Processing complete!"
        self.status_str.set(self.status)
        self.root.update()
        self.ImageDisplay(1, self.folderName, 1)

    def PredictDLC(self, config, input_file, output, atlas, sensory_match, sensory_path, model, num_images, mat_save, threshold,
                   mask_generate, haveMasks, git_repo_base, region_labels):
        self.status = "Processing..."
        self.status_str.set(self.status)
        self.root.update()
        if mask_generate and not haveMasks:
            predictRegion(input_file, num_images, model, output, mat_save, threshold, mask_generate, git_repo_base,
                          region_labels)
        DLCPredict(config, input_file, output, atlas, sensory_match, sensory_path,
                   mat_save, threshold, git_repo_base, region_labels)
        saveFolderName = output
        if not atlas:
            self.folderName = os.path.join(saveFolderName, "output_overlay")
        elif atlas:
            self.folderName = os.path.join(saveFolderName, "dlc_output")
        config_project(input_file, saveFolderName, 'test', config=config, atlas=atlas, sensory_match=sensory_match,
                       mat_save=mat_save, threshold=threshold, model=model, region_labels=region_labels)
        self.status = "Processing complete!"
        self.status_str.set(self.status)
        self.root.update()
        self.ImageDisplay(1, self.folderName, 1)


def gui():
    Gui().root.mainloop()