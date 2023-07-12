"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the Creative Commons Attribution 4.0 International License (see LICENSE for details)
"""

import fnmatch
import glob
import os
import numpy as np
import skimage.io as io
from tkinter import *  # Python 3.x
from tkinter import filedialog

from PIL import Image, ImageTk, ImageDraw

from mesonet.train_model import trainModel
from mesonet.utils import config_project, find_git_repo
from mesonet.dlc_predict import DLCPrep, DLCLabel, DLCTrain, DLC_edit_bodyparts
from mesonet.mask_functions import inpaintMask


class GuiTrain:
    """
    The main GUI interface for training new U-Net and DLC models for use in MesoNet.
    """

    DEFAULT_PEN_SIZE = 20
    DEFAULT_COLOUR = "white"
    DEFAULT_MODEL_NAME = "my_unet.hdf5"
    DEFAULT_TASK = "MesoNet"
    DEFAULT_NAME = "Labeler"

    def __init__(self):
        self.root_train = Tk()
        self.root_train.resizable(False, False)
        self.Title = self.root_train.title("MesoNet Trainer")

        self.status = 'Please select a folder with brain images at "Input Folder".'
        self.status_str = StringVar(self.root_train, value=self.status)

        self.cwd = os.getcwd()
        self.logName = self.cwd
        self.git_repo_base = find_git_repo()
        self.folderName = self.cwd
        self.saveFolderName = self.cwd
        self.model_name = self.DEFAULT_MODEL_NAME
        self.dlc_folder = self.cwd
        self.task = self.DEFAULT_TASK
        self.name = self.DEFAULT_NAME
        self.bodyparts = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        self.config_path = ""
        self.steps_per_epoch = 300
        self.epochs = 60

        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOUR
        self.cv_dim = 512
        self.canvas = Canvas(self.root_train, width=self.cv_dim, height=self.cv_dim)
        self.canvas.grid(row=5, column=0, columnspan=4, rowspan=9, sticky=N + S + W)
        self.mask = Image.new("L", (self.cv_dim, self.cv_dim))
        self.draw = ImageDraw.Draw(self.mask)

        # Set file input and output
        self.fileEntryLabel = Label(self.root_train, text="Input folder")
        self.fileEntryLabel.grid(row=0, column=0, sticky=E + W)
        self.fileEntryButton = Button(
            self.root_train, text="Browse...", command=lambda: self.OpenFile(0)
        )
        self.folderName_str = StringVar(self.root_train, value=self.folderName)
        self.fileEntryButton.grid(row=0, column=2, sticky=E)
        self.fileEntryBox = Entry(
            self.root_train, textvariable=self.folderName_str, width=60
        )
        self.fileEntryBox.grid(row=0, column=1, padx=5, pady=5)

        self.fileSaveLabel = Label(self.root_train, text="Save folder")
        self.fileSaveLabel.grid(row=1, column=0, sticky=E + W)
        self.fileSaveButton = Button(
            self.root_train, text="Browse...", command=lambda: self.OpenFile(1)
        )
        self.saveFolderName_str = StringVar(self.root_train, value=self.saveFolderName)
        self.fileSaveButton.grid(row=1, column=2, sticky=E)
        self.fileSaveBox = Entry(
            self.root_train, textvariable=self.saveFolderName_str, width=60
        )
        self.fileSaveBox.grid(row=1, column=1, padx=5, pady=5)

        self.logSaveLabel = Label(self.root_train, text="Log folder")
        self.logSaveLabel.grid(row=2, column=0, sticky=E + W)
        self.logSaveButton = Button(
            self.root_train, text="Browse...", command=lambda: self.OpenFile(2)
        )
        self.logName_str = StringVar(self.root_train, value=self.logName)
        self.logSaveButton.grid(row=2, column=2, sticky=E)
        self.logSaveBox = Entry(
            self.root_train, textvariable=self.logName_str, width=60
        )
        self.logSaveBox.grid(row=2, column=1, padx=5, pady=5)

        self.line_width_str = StringVar(self.root_train, value=self.line_width)
        self.lineWidthLabel = Label(self.root_train, text="Brush size")
        self.lineWidthLabel.grid(row=2, column=4, sticky=E + W)
        self.lineWidthBox = Entry(
            self.root_train, textvariable=self.line_width_str, width=20
        )
        self.lineWidthBox.grid(row=2, column=5)

        self.model_name_str = StringVar(self.root_train, value=self.model_name)
        self.modelNameLabel = Label(self.root_train, text="Model name")
        self.modelNameLabel.grid(row=3, column=4, sticky=E + W)
        self.modelNameBox = Entry(
            self.root_train, textvariable=self.model_name_str, width=20
        )
        self.modelNameBox.grid(row=3, column=5)

        self.dlc_folder_str = StringVar(self.root_train, value=self.dlc_folder)
        self.dlcFolderLabel = Label(self.root_train, text="DLC Folder")
        self.dlcFolderLabel.grid(row=3, column=0, sticky=E + W)
        self.dlcFolderButton = Button(
            self.root_train, text="Browse...", command=lambda: self.OpenFile(3)
        )
        self.dlcFolderButton.grid(row=3, column=2, sticky=E)
        self.dlcFolderBox = Entry(
            self.root_train, textvariable=self.dlc_folder_str, width=60
        )
        self.dlcFolderBox.grid(row=3, column=1, padx=5, pady=5)

        # Saving and training setup buttons
        self.saveButton = Button(
            self.root_train,
            text="Save current mask to file",
            command=lambda: self.mask_save(self.saveFolderName, self.j),
        )
        self.saveButton.grid(
            row=9, column=4, columnspan=2, padx=2, sticky=N + S + W + E
        )

        self.trainButton = Button(
            self.root_train,
            text="Train U-net model",
            command=lambda: self.trainModelGUI(
                self.saveFolderName,
                os.path.join(self.git_repo_base, "models", self.modelNameBox.get()),
                self.logName,
                self.git_repo_base,
                int(self.stepEpochsBox.get()),
                int(self.epochsBox.get()),
            ),
        )
        self.trainButton.grid(
            row=10, column=4, columnspan=2, padx=2, sticky=N + S + W + E
        )

        # Setup DLC training options
        self.task_str = StringVar(self.root_train, value=self.task)
        self.taskLabel = Label(self.root_train, text="Task")
        self.taskLabel.grid(row=0, column=4, sticky=E + W)
        self.taskBox = Entry(self.root_train, textvariable=self.task_str, width=20)
        self.taskBox.grid(row=0, column=5)

        self.name_str = StringVar(self.root_train, value=self.name)
        self.nameLabel = Label(self.root_train, text="Name")
        self.nameLabel.grid(row=1, column=4, sticky=E + W)
        self.nameBox = Entry(self.root_train, textvariable=self.name_str, width=20)
        self.nameBox.grid(row=1, column=5)

        self.epochs_str = StringVar(self.root_train, value=self.epochs)
        self.epochsLabel = Label(self.root_train, text="U-Net epochs")
        self.epochsLabel.grid(row=4, column=4, sticky=E + W)
        self.epochsBox = Entry(self.root_train, textvariable=self.epochs_str, width=20)
        self.epochsBox.grid(row=4, column=5)

        self.step_epochs_str = StringVar(self.root_train, value=self.steps_per_epoch)
        self.stepEpochsLabel = Label(self.root_train, text="Steps per epoch")
        self.stepEpochsLabel.grid(row=5, column=4, sticky=E + W)
        self.stepEpochsBox = Entry(
            self.root_train, textvariable=self.step_epochs_str, width=20
        )
        self.stepEpochsBox.grid(row=5, column=5)

        self.displayiters = 100
        self.displayiters_str = StringVar(self.root_train, value=self.displayiters)
        self.displayitersLabel = Label(self.root_train, text="Display iters")
        self.displayitersLabel.grid(row=6, column=4, sticky=E + W)
        self.displayitersBox = Entry(
            self.root_train, textvariable=self.displayiters_str, width=20
        )
        self.displayitersBox.grid(row=6, column=5)

        self.saveiters = 1000
        self.saveiters_str = StringVar(self.root_train, value=self.saveiters)
        self.saveitersLabel = Label(self.root_train, text="Save iters")
        self.saveitersLabel.grid(row=7, column=4, sticky=E + W)
        self.saveitersBox = Entry(
            self.root_train, textvariable=self.saveiters_str, width=20
        )
        self.saveitersBox.grid(row=7, column=5)

        self.maxiters = 30000
        self.maxiters_str = StringVar(self.root_train, value=self.maxiters)
        self.maxitersLabel = Label(self.root_train, text="Max iters")
        self.maxitersLabel.grid(row=8, column=4, sticky=E + W)
        self.maxitersBox = Entry(
            self.root_train, textvariable=self.maxiters_str, width=20
        )
        self.maxitersBox.grid(row=8, column=5)

        # Generate DLC config file
        self.dlcConfigButton = Button(
            self.root_train,
            text="Generate DLC config file",
            command=lambda: self.getDLCConfig(
                self.taskBox.get(), self.nameBox.get(), self.folderName, self.dlc_folder
            ),
        )
        self.dlcConfigButton.grid(row=11, column=4, columnspan=2, sticky=N + S + W + E)

        self.dlcLabelButton = Button(
            self.root_train,
            text="Label brain images\nwith landmarks",
            command=lambda: DLCLabel(self.config_path),
        )
        self.dlcLabelButton.grid(row=12, column=4, columnspan=2, sticky=N + S + W + E)

        self.dlcTrainButton = Button(
            self.root_train,
            text="Train DLC model",
            command=lambda: DLCTrain(
                self.config_path,
                self.displayitersBox.get(),
                self.saveitersBox.get(),
                self.maxitersBox.get(),
            ),
        )
        self.dlcTrainButton.grid(row=13, column=4, columnspan=2, sticky=N + S + W + E)

        # Image controls
        # Buttons below will only display if an image is displayed
        self.nextButton = Button(
            self.root_train, text="->", command=lambda: self.forward(None)
        )
        self.nextButton.grid(row=14, column=2, columnspan=1)
        self.previousButton = Button(
            self.root_train, text="<-", command=lambda: self.backward(None)
        )
        self.previousButton.grid(row=14, column=0, columnspan=1)

        # Bind right and left arrow keys to forward/backward controls
        self.root_train.bind("<Right>", self.forward)
        self.root_train.bind("<Left>", self.backward)

        self.paint_setup()

    def OpenFile(self, openOrSave):
        if openOrSave == 0:
            newFolderName = filedialog.askdirectory(
                initialdir=self.cwd,
                title="Choose folder containing the brain images you want to analyze",
            )
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.folderName_str.set(newFolderName)
                self.folderName = newFolderName
                self.ImageDisplay(1, self.folderName, 1)
                self.status = (
                    'Please select a folder to save your images to at "Save Folder".'
                )
                self.status_str.set(self.status)
                self.root_train.update()
            except:
                print("No image file selected!")
        elif openOrSave == 1:
            newSaveFolderName = filedialog.askdirectory(
                initialdir=self.cwd, title="Choose folder for saving files"
            )
            # Using try in case user types in unknown file or closes without choosing a file.
            try:
                self.saveFolderName_str.set(newSaveFolderName)
                self.saveFolderName = newSaveFolderName
                self.status = "Save folder selected! Choose an option on the right to begin your analysis."
                self.status_str.set(self.status)
                self.root_train.update()
            except:
                print("No save file selected!")
                self.status = "No save file selected!"
                self.status_str.set(self.status)
                self.root_train.update()
        elif openOrSave == 2:
            newLogName = filedialog.askdirectory(
                initialdir=self.cwd,
                title="Choose folder for saving model training logs",
            )
            try:
                self.logName_str.set(newLogName)
                self.logName = newLogName
                self.root_train.update()
            except:
                print("No log folder selected!")
        elif openOrSave == 3:
            newDLCFolder = filedialog.askdirectory(
                initialdir=self.cwd, title="Choose folder for DLC project"
            )
            try:
                self.dlc_folder_str.set(newDLCFolder)
                self.dlc_folder = newDLCFolder
                self.root_train.update()
            except:
                print("No DLC folder selected!")

    def ImageDisplay(self, delta, folderName, reset):
        # Set up canvas on which images will be displayed
        self.imgDisplayed = 1
        self.root_train.update()
        if reset == 1:
            self.j = -1
        self.j += delta
        if glob.glob(os.path.join(folderName, "*_mask_segmented.png")):
            file_list = glob.glob(os.path.join(folderName, "*_mask_segmented.png"))
        elif glob.glob(os.path.join(folderName, "*_mask.png")):
            file_list = glob.glob(os.path.join(folderName, "*_mask.png"))
        else:
            file_list = glob.glob(os.path.join(folderName, "*.png"))
        self.picLen = len(file_list)
        if self.j > self.picLen - 1:
            self.j = 0
        if self.j <= -1:
            self.j = self.picLen - 1
        if delta != 0:
            for file in file_list:
                if (
                    fnmatch.fnmatch(
                        file,
                        os.path.join(
                            folderName, "{}_mask_segmented.png".format(self.j)
                        ),
                    )
                    or fnmatch.fnmatch(
                        file, os.path.join(folderName, "{}.png".format(self.j))
                    )
                    or fnmatch.fnmatch(
                        file, os.path.join(folderName, "{}_mask.png".format(self.j))
                    )
                ):
                    self.imageFileName = os.path.basename(file)
                    image = os.path.join(folderName, file)
                    image_orig = Image.open(image)
                    self.image_resize = image_orig.resize((512, 512))
                    image_disp = ImageTk.PhotoImage(self.image_resize)
                    self.canvas.create_image(256, 256, image=image_disp)
                    label = Label(image=image_disp)
                    label.image = image_disp
                    self.root_train.update()
        imageName = StringVar(self.root_train, value=self.imageFileName)
        imageNum = "Image {}/{}".format(self.j + 1, self.picLen)
        imageNumPrep = StringVar(self.root_train, value=imageNum)
        imageNameLabel = Label(self.root_train, textvariable=imageName)
        imageNameLabel.grid(row=4, column=0, columnspan=1)
        imageNumLabel = Label(self.root_train, textvariable=imageNumPrep)
        imageNumLabel.grid(row=4, column=2, columnspan=1)

    def forward(self, event):
        self.ImageDisplay(1, self.folderName, 0)
        self.mask = Image.new("L", (self.cv_dim, self.cv_dim))
        self.draw = ImageDraw.Draw(self.mask)

    def backward(self, event):
        self.ImageDisplay(-1, self.folderName, 0)
        self.mask = Image.new("L", (self.cv_dim, self.cv_dim))
        self.draw = ImageDraw.Draw(self.mask)

    def paint_setup(self):
        self.old_x, self.old_y = None, None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        """
        https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06
        :param event:
        :return:
        """
        line_width = int(self.lineWidthBox.get())
        paint_color = self.color
        if self.old_x and self.old_y:
            self.canvas.create_oval(
                event.x - (line_width / 2),
                event.y - (line_width / 2),
                event.x + (line_width / 2),
                event.y + (line_width / 2),
                fill=paint_color,
                outline=paint_color,
            )
            self.draw.ellipse(
                (
                    event.x - (line_width / 2),
                    event.y - (line_width / 2),
                    event.x + (line_width / 2),
                    event.y + (line_width / 2),
                ),
                fill=paint_color,
                outline=paint_color,
            )
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def mask_save(self, mask_folder, img_name):
        if not os.path.isdir(os.path.join(mask_folder, "image")):
            os.mkdir(os.path.join(mask_folder, "image"))
        if not os.path.isdir(os.path.join(mask_folder, "label")):
            os.mkdir(os.path.join(mask_folder, "label"))

        self.image_resize.save(
            os.path.join(mask_folder, "image", "{}.png".format(img_name))
        )
        mask_cv2 = np.array(self.mask)
        mask_cv2 = inpaintMask(mask_cv2)
        io.imsave(
            os.path.join(mask_folder, "label", "{}.png".format(img_name)), mask_cv2
        )
        # self.mask.save(os.path.join(mask_folder, "label", "{}.png".format(img_name)))

    def trainModelGUI(
        self,
        mask_folder,
        model_name,
        log_folder,
        git_repo_base,
        steps_per_epoch,
        epochs,
    ):
        trainModel(
            mask_folder, model_name, log_folder, git_repo_base, steps_per_epoch, epochs
        )
        config_project(mask_folder, log_folder, "train", model_name=model_name)

    def getDLCConfig(self, project_name, your_name, img_path, output_dir_base):
        config_path = DLCPrep(project_name, your_name, img_path, output_dir_base)
        print(config_path)
        self.config_path = config_path
        DLC_edit_bodyparts(self.config_path, self.bodyparts)


def gui():
    GuiTrain().root_train.mainloop()
