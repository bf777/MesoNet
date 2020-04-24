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

from PIL import Image, ImageTk, ImageDraw

from mesonet.train_model import trainModel
from mesonet.gui_class import Gui as gui


class GuiTrain:
    def __init__(self):
        self.root_train = Tk()
        self.root_train.resizable(False, False)

        self.cwd = os.getcwd()
        self.folderName = self.cwd
        self.saveFolderName = self.cwd

        self.line_width = 20
        self.color = 'white'
        self.cv_dim = 512
        self.canvas = Canvas(self.root_train, width=self.cv_dim, height=self.cv_dim)
        self.canvas.grid(row=6, column=0, columnspan=4, rowspan=8, sticky=N + S + W)
        self.mask = Image.new("L", (self.cv_dim, self.cv_dim))
        self.draw = ImageDraw.Draw(self.mask)

        self.line_width_str = StringVar(self.root_train, value=self.line_width)
        self.lineWidthBox = Entry(self.root_train, textvariable=self.line_width_str, width=60)
        self.lineWidthBox.grid(row=2, column=1, padx=5, pady=5)

        # Set file input and output
        self.fileEntryLabel = Label(self.root_train, text="Input folder")
        self.fileEntryLabel.grid(row=0, column=0, sticky=E + W)
        self.fileEntryButton = Button(self.root_train, text="Browse...", command=lambda: gui.OpenFile(gui, 0))

        self.folderName_str = StringVar(self.root_train, value=self.folderName)
        self.fileEntryButton.grid(row=0, column=2, sticky=E)
        self.fileEntryBox = Entry(self.root_train, textvariable=self.folderName_str, width=60)
        self.fileEntryBox.grid(row=0, column=1, padx=5, pady=5)

        self.fileSaveLabel = Label(self.root_train, text="Save folder")
        self.fileSaveLabel.grid(row=1, column=0, sticky=E + W)
        self.fileSaveButton = Button(self.root_train, text="Browse...", command=lambda: gui.OpenFile(gui, 1))

        self.saveFolderName_str = StringVar(self.root_train, value=self.saveFolderName)
        self.fileSaveButton.grid(row=1, column=2, sticky=E)
        self.fileSaveBox = Entry(self.root_train, textvariable=self.saveFolderName_str, width=60)
        self.fileSaveBox.grid(row=1, column=1, padx=5, pady=5)

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
        self.line_width = self.lineWidthBox.get()
        paint_color = self.color
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=self.line_width, fill=paint_color,
                                    capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line(self.old_x, self.old_y, event.x, event.y, width=self.line_width,
                           fill=paint_color)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def mask_save(self, mask_folder, img_name):
        self.mask.save(os.path.join(mask_folder, img_name))


def gui():
    GuiTrain().root_train.mainloop()
