"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""

import fnmatch
import glob
import os
import aggdraw
from tkinter import *  # Python 3.x
from tkinter import filedialog

from PIL import Image, ImageTk

from mesonet.train_model import trainModel
from mesonet.utils import config_project, find_git_repo


class GuiTrain:

    DEFAULT_PEN_SIZE = 20
    DEFAULT_COLOUR = 'white'

    def __init__(self):

        self.root_train = Tk()
        self.root_train.resizable(False, False)

        self.status = 'Please select a folder with brain images at "Input Folder".'
        self.status_str = StringVar(self.root_train, value=self.status)

        self.cwd = os.getcwd()
        self.logName = self.cwd
        self.git_repo_base = find_git_repo()
        self.folderName = self.cwd
        self.saveFolderName = self.cwd
        self.model_name = 'my_unet.hdf5'

        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOUR
        self.cv_dim = 512
        self.canvas = Canvas(self.root_train, width=self.cv_dim, height=self.cv_dim)
        self.canvas.grid(row=6, column=0, columnspan=4, rowspan=8, sticky=N + S + W)
        self.mask = Image.new("L", (self.cv_dim, self.cv_dim))
        self.draw = aggdraw.Draw(self.mask)

        self.line_width_str = StringVar(self.root_train, value=self.line_width)
        self.lineWidthLabel = Label(self.root_train, text="Brush size")
        self.lineWidthLabel.grid(row=3, column=0, sticky=E + W)
        self.lineWidthBox = Entry(self.root_train, textvariable=self.line_width_str, width=60)
        self.lineWidthBox.grid(row=3, column=1, padx=5, pady=5)

        # Set file input and output
        self.fileEntryLabel = Label(self.root_train, text="Input folder")
        self.fileEntryLabel.grid(row=0, column=0, sticky=E + W)
        self.fileEntryButton = Button(self.root_train, text="Browse...", command=lambda: self.OpenFile(0))
        self.folderName_str = StringVar(self.root_train, value=self.folderName)
        self.fileEntryButton.grid(row=0, column=2, sticky=E)
        self.fileEntryBox = Entry(self.root_train, textvariable=self.folderName_str, width=60)
        self.fileEntryBox.grid(row=0, column=1, padx=5, pady=5)

        self.fileSaveLabel = Label(self.root_train, text="Save folder")
        self.fileSaveLabel.grid(row=1, column=0, sticky=E + W)
        self.fileSaveButton = Button(self.root_train, text="Browse...", command=lambda: self.OpenFile(1))
        self.saveFolderName_str = StringVar(self.root_train, value=self.saveFolderName)
        self.fileSaveButton.grid(row=1, column=2, sticky=E)
        self.fileSaveBox = Entry(self.root_train, textvariable=self.saveFolderName_str, width=60)
        self.fileSaveBox.grid(row=1, column=1, padx=5, pady=5)

        self.logSaveLabel = Label(self.root_train, text="Log folder")
        self.logSaveLabel.grid(row=2, column=0, sticky=E + W)
        self.logSaveButton = Button(self.root_train, text="Browse...", command=lambda: self.OpenFile(1))
        self.logName_str = StringVar(self.root_train, value=self.logName)
        self.logSaveButton.grid(row=2, column=2, sticky=E)
        self.logSaveBox = Entry(self.root_train, textvariable=self.saveFolderName_str, width=60)
        self.logSaveBox.grid(row=2, column=1, padx=5, pady=5)

        # Image controls
        # Buttons below will only display if an image is displayed
        self.nextButton = Button(self.root_train, text="->", command=lambda: self.ImageDisplay(1, self.folderName, 0))
        self.nextButton.grid(row=14, column=2, columnspan=1)
        self.previousButton = Button(self.root_train, text="<-", command=lambda: self.ImageDisplay(-1, self.folderName,
                                                                                                   0))
        self.previousButton.grid(row=14, column=0, columnspan=1)

        # Bind right and left arrow keys to forward/backward controls
        self.root_train.bind('<Right>', self.forward)
        self.root_train.bind('<Left>', self.backward)

        self.saveButton = Button(self.root_train, text="Save all masks to file",
                                 command=lambda: self.mask_save(self.saveFolderName, self.j))
        self.saveButton.grid(row=13, column=3,  padx=2, sticky=N + S + W + E)

        self.trainButton = Button(self.root_train, text="Train U-net model",
                                  command=lambda: self.trainModelGUI(self.saveFolderName, self.model_name, self.logName,
                                                                     self.git_repo_base))
        self.trainButton.grid(row=14, column=3, padx=2, sticky=N + S + W + E)
        self.paint_setup()

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
                self.root_train.update()
            except:
                print("No image file selected!")
        elif openOrSave == 1:
            newSaveFolderName = filedialog.askdirectory(initialdir=self.cwd,
                                                        title="Choose folder for saving files")
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
            newLogName = filedialog.askdirectory(initialdir=self.cwd,
                                                     title="Choose folder for saving model training logs")
            try:
                self.logName_str.set(newLogName)
                self.logName = newLogName
                self.root_train.update()
            except:
                print("No sensory image file selected!")

    def ImageDisplay(self, delta, folderName, reset):
        # Set up canvas on which images will be displayed
        self.imgDisplayed = 1
        self.root_train.update()
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
                    self.image_resize = image_orig.resize((512, 512))
                    image_disp = ImageTk.PhotoImage(self.image_resize)
                    self.canvas.create_image(256, 256, image=image_disp)
                    label = Label(image=image_disp)
                    label.image = image_disp
                    self.root_train.update()
        imageName = StringVar(self.root_train, value=self.imageFileName)
        imageNum = 'Image {}/{}'.format(self.j + 1, self.picLen)
        imageNumPrep = StringVar(self.root_train, value=imageNum)
        imageNameLabel = Label(self.root_train, textvariable=imageName)
        imageNameLabel.grid(row=5, column=0, columnspan=1)
        imageNumLabel = Label(self.root_train, textvariable=imageNumPrep)
        imageNumLabel.grid(row=5, column=2, columnspan=1)

    def forward(self, event):
        self.ImageDisplay(1, self.folderName, 0)

    def backward(self, event):
        self.ImageDisplay(-1, self.folderName, 0)

    def paint_setup(self):
        self.old_x, self.old_y = None, None
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def paint(self, event):
        """
        https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06
        :param event:
        :return:
        """
        line_width = int(self.lineWidthBox.get())
        paint_color = self.color
        pen = aggdraw.Pen(color=paint_color, width=line_width)
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=line_width, fill=paint_color,
                                    capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line((self.old_x, self.old_y, event.x, event.y), pen)
            self.draw.flush()
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def mask_save(self, mask_folder, img_name):
        if not os.path.isdir(os.path.join(mask_folder, 'image')):
            os.mkdir(os.path.join(mask_folder, 'image'))
        if not os.path.isdir(os.path.join(mask_folder, 'label')):
            os.mkdir(os.path.join(mask_folder, 'label'))

        self.image_resize.save(os.path.join(mask_folder, "image", "{}.png".format(img_name)))
        self.mask.save(os.path.join(mask_folder, "label", "{}.png".format(img_name)))

    def trainModelGUI(self, mask_folder, model_name, log_folder, git_repo_base):
        trainModel(mask_folder, model_name, log_folder, git_repo_base)


def gui():
    GuiTrain().root_train.mainloop()
