"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
from mesonet.predict_regions import predictRegion
from mesonet.dlc_predict import DLCPredict, DLCPredictBehavior
from mesonet.utils import config_project, find_git_repo
from tkinter import *  # Python 3.x
from tkinter import filedialog
from PIL import Image, ImageTk

import os
import fnmatch
import glob
from os.path import join
from sys import platform

# The main window of the app
root = Tk()
root.resizable(False, False)

cwd = os.getcwd()
folderName = cwd
BfolderName = cwd
saveFolderName = ''
saveBFolderName = ''
threshold = 0.001
haveMasks = False
# status = 'Please select a folder with brain images at "Input Folder".'

j = -1
delta = 0
imgDisplayed = 0

config_dir = 'dlc'
model_dir = 'models'

git_repo_base = find_git_repo()

config_path = os.path.join(git_repo_base, config_dir, 'config.yaml')
behavior_config_path = os.path.join(git_repo_base, config_dir, 'behavior',' config.yaml')
model_top_dir = os.path.join(git_repo_base, model_dir)


def OpenFile(openOrSave):
    if openOrSave == 0:
        newFolderName = filedialog.askdirectory(initialdir=cwd,
                                                title="Choose folder containing the brain images you want to analyze")
        # Using try in case user types in unknown file or closes without choosing a file.
        try:
            folderName_str.set(newFolderName)
            global folderName
            # global status
            # status = 'Brain images found! Now select a folder to save outputs using the "Save Folder" box.'
            folderName = newFolderName
            ImageDisplay(1, folderName, 1)
            root.update()
        except:
            print("No image file selected!")
    elif openOrSave == 1:
        newSaveFolderName = filedialog.askdirectory(initialdir=cwd,
                                                    title="Choose folder for saving files")
        # Using try in case user types in unknown file or closes without choosing a file.
        try:
            saveFolderName_str.set(newSaveFolderName)
            global saveFolderName
            saveFolderName = newSaveFolderName
            predictAllImButton.config(state='normal')
            generateMaskButton.config(state='normal')
            predictDLCButton.config(state='normal')
            saveMatFileCheck.config(state='normal')
            predictLandmarksButton.config(state='normal')
            sensoryMapCheck.config(state='normal')
            # status = "Save folder selected! Choose an option on the right to begin your analysis."
            root.update()
        except:
            print("No save file selected!")
            # status = "No save file selected!"
            # root.update()


def OpenBFile(openOrSave):
    if openOrSave == 0:
        newBFolderName = filedialog.askdirectory(initialdir=cwd,
                                                title="Choose folder containing the brain images you want to analyze")
        # Using try in case user types in unknown file or closes without choosing a file.
        try:
            BfolderName_str.set(newBFolderName)
            global BfolderName
            # global status
            # status = 'Brain images found! Now select a folder to save outputs using the "Save Folder" box.'
            BfolderName = newBFolderName
            root.update()
        except:
            print("No image file selected!")

    elif openOrSave == 1:
        newSaveBFolderName = filedialog.askdirectory(initialdir=cwd,
                                                    title="Choose folder for saving files")
        # Using try in case user types in unknown file or closes without choosing a file.
        try:
            saveBFolderName_str.set(newSaveBFolderName)
            global saveBFolderName
            saveBFolderName = newSaveBFolderName
            predictBehaviourButton.config(state='normal')
            root.update()
        except:
            print("No save file selected!")
            # status = "No save file selected!"
            # root.update()


def ImageDisplay(delta, folderName, reset):
    # Set up canvas on which images will be displayed
    global imgDisplayed
    imgDisplayed = 1
    root.update()
    global j
    if reset == 1:
        j = -1
    j += delta
    fileList = glob.glob(os.path.join(folderName, '*.png'))
    global picLen
    picLen = len(fileList)
    if j > picLen - 1:
        j = 0
    if j <= -1:
        j = picLen - 1
    if delta != 0:
        for file in fileList:
            if fnmatch.fnmatch(file, os.path.join(folderName, "{}_mask_segmented.png".format(j))) or \
                    fnmatch.fnmatch(file, os.path.join(folderName, "{}.png".format(j))) or \
                    fnmatch.fnmatch(file, os.path.join(folderName, "{}_mask.png".format(j))):
                global imageFileName
                imageFileName = os.path.basename(file)
                image = os.path.join(folderName, file)
                image_orig = Image.open(image)
                image_resize = image_orig.resize((512, 512))
                image_disp = ImageTk.PhotoImage(image_resize)
                canvas.create_image(256, 256, image=image_disp)
                label = Label(image=image_disp)
                label.image = image_disp
                root.update()
    imageName = StringVar(root, value=imageFileName)
    imageNum = 'Image {}/{}'.format(j + 1, picLen)
    imageNumPrep = StringVar(root, value=imageNum)
    imageNameLabel = Label(root, textvariable=imageName)
    imageNameLabel.grid(row=4, column=0, columnspan=1)
    imageNumLabel = Label(root, textvariable=imageNumPrep)
    imageNumLabel.grid(row=4, column=2, columnspan=1)


def FindModelInFolder():
    modelSelect = []
    for file in os.listdir(model_top_dir):
        if fnmatch.fnmatch(file, "*.hdf5"):
            modelSelect.append(file)

    modelLabel = Label(root, text="If using U-net, select a model\nto analyze the brain regions:")
    modelListBox = Listbox(root)
    modelLabel.grid(row=4, column=4, sticky=S)
    modelListBox.grid(row=5, column=4, sticky=N)
    for item in modelSelect:
        modelListBox.insert(END, item)
    if len(modelSelect) > 0:
        def onselect(evt):
            global model
            w = evt.widget
            selected = int(w.curselection()[0])
            model = modelListBox.get(selected)
            model = os.path.join(git_repo_base, model_dir, model)
            root.update()
        modelListBox.bind('<<ListboxSelect>>', onselect)


def forward(event):
    ImageDisplay(1, folderName, 0)


def backward(event):
    ImageDisplay(-1, folderName, 0)


def PredictRegions(input_file, num_images, model, output, mat_save, threshold, mask_generate):
    # global status
    # status = "Processing..."
    root.update()
    predictRegion(input_file, num_images, model, output, mat_save, threshold, mask_generate)
    saveFolderName = output
    global folderName
    if mask_generate:
        folderName = os.path.join(saveFolderName, "output_mask")
        global haveMasks
        haveMasks = True
    else:
        folderName = saveFolderName
    # status = "Processing complete!"
    root.update()
    ImageDisplay(1, folderName, 1)


def PredictDLC(config, input_file, output, atlas, sensory_match, model, num_images, mat_save, threshold, mask_generate,
               haveMasks, git_repo_base):
    global status
    status = "Processing..."
    root.update()
    if mask_generate and not haveMasks:
        predictRegion(input_file, num_images, model, output, mat_save, threshold, True)
    DLCPredict(config, input_file, output, atlas, sensory_match, mat_save, threshold, git_repo_base)
    saveFolderName = output
    global folderName
    if not atlas:
        folderName = os.path.join(saveFolderName, "output_overlay")
    elif atlas:
        folderName = os.path.join(saveFolderName, "dlc_output")
    config_project(input_file, saveFolderName, 'test', config=config, atlas=atlas, sensory_match=sensory_match,
                   mat_save=mat_save, threshold=threshold, model=model)
    root.update()
    ImageDisplay(1, folderName, 1)


Title = root.title("MesoNet Analyzer")

canvas = Canvas(root, width=512, height=512)
canvas.grid(row=5, column=0, columnspan=4, rowspan=8, sticky=N + S + W)

# Render model selector listbox
FindModelInFolder()

# Set file input and output
fileEntryLabel = Label(root, text="Input folder")
fileEntryLabel.grid(row=0, column=0, sticky=E + W)
fileEntryButton = Button(root, text="Browse...", command=lambda: OpenFile(0))

folderName_str = StringVar(root, value=folderName)
fileEntryButton.grid(row=0, column=2, sticky=E)
fileEntryBox = Entry(root, textvariable=folderName_str, width=60)
fileEntryBox.grid(row=0, column=1, padx=5, pady=5)

fileSaveLabel = Label(root, text="Save folder")
fileSaveLabel.grid(row=1, column=0, sticky=E + W)
fileSaveButton = Button(root, text="Browse...", command=lambda: OpenFile(1))

saveFolderName_str = StringVar(root, value=saveFolderName)
fileSaveButton.grid(row=1, column=2, sticky=E)
fileSaveBox = Entry(root, textvariable=saveFolderName_str, width=60)
fileSaveBox.grid(row=1, column=1, padx=5, pady=5)

# Set behavioural data files
BfileEntryLabel = Label(root, text="Behavior Input folder")
BfileEntryLabel.grid(row=2, column=0, sticky=E + W)
BfileEntryButton = Button(root, text="Browse...", command=lambda: OpenBFile(0))

BfolderName_str = StringVar(root, value=folderName)
BfileEntryButton.grid(row=2, column=2, sticky=E)
BfileEntryBox = Entry(root, textvariable=BfolderName_str, width=60)
BfileEntryBox.grid(row=2, column=1, padx=5, pady=5)

BfileSaveLabel = Label(root, text="Behavior Save folder")
BfileSaveLabel.grid(row=3, column=0, sticky=E + W)
BfileSaveButton = Button(root, text="Browse...", command=lambda: OpenBFile(1))

saveBFolderName_str = StringVar(root, value=saveBFolderName)
BfileSaveButton.grid(row=3, column=2, sticky=E)
BfileSaveBox = Entry(root, textvariable=saveBFolderName_str, width=60)
BfileSaveBox.grid(row=3, column=1, padx=5, pady=5)

# Image controls
# Buttons below will only display if an image is displayed
nextButton = Button(root, text="->", command=lambda: ImageDisplay(1, folderName, 0))
nextButton.grid(row=13, column=2, columnspan=1)
previousButton = Button(root, text="<-", command=lambda: ImageDisplay(-1, folderName, 0))
previousButton.grid(row=13, column=0, columnspan=1)

# Bind right and left arrow keys to forward/backward controls
root.bind('<Right>', forward)
root.bind('<Left>', backward)

# Buttons for making predictions
# Buttons below will only be active if a save file has been selected
mat_save = IntVar()
atlas = IntVar()
sensory_align = IntVar()
saveMatFileCheck = Checkbutton(root, text="Save predicted regions\nas .mat files", variable=mat_save)
saveMatFileCheck.grid(row=6, column=4, padx=2, sticky=N+S+W)
sensoryMapCheck = Checkbutton(root, text="Align using sensory map", variable=sensory_align)
sensoryMapCheck.grid(row=7, column=4, padx=2, sticky=N+S+W)
generateMaskButton = Button(root, text="Get boundaries of brain\nusing U-net",
                            command=lambda: PredictRegions(folderName, picLen,
                                                           os.path.join(model_top_dir, 'unet_bundary.hdf5'),
                                                           saveFolderName, int(mat_save.get()),
                                                           threshold, True))
generateMaskButton.grid(row=8, column=4, padx=2, sticky=N + S + W + E)
predictLandmarksButton = Button(root, text="Predict landmark locations",
                          command=lambda: PredictDLC(config_path, folderName, saveFolderName, True,
                                                     int(sensory_align.get()),
                                                     os.path.join(model_top_dir, 'unet_bundary.hdf5'), picLen,
                                                     int(mat_save.get()), threshold, False, haveMasks, git_repo_base))
predictLandmarksButton.grid(row=9, column=4, padx=2, sticky=N + S + W + E)
predictDLCButton = Button(root, text="Predict brain regions\nusing landmarks",
                          command=lambda: PredictDLC(config_path, folderName, saveFolderName, False,
                                                     int(sensory_align.get()),
                                                     os.path.join(model_top_dir, 'unet_bundary.hdf5'), picLen,
                                                     int(mat_save.get()), threshold, True, haveMasks, git_repo_base))
predictDLCButton.grid(row=10, column=4, padx=2, sticky=N + S + W + E)
predictAllImButton = Button(root, text="Predict brain regions directly\nusing pretrained U-net model",
                            command=lambda: PredictRegions(folderName, picLen, model, saveFolderName,
                                                           int(mat_save.get()), threshold, False))
predictAllImButton.grid(row=11, column=4, padx=2, sticky=N + S + W + E)
predictBehaviourButton = Button(root, text="Predict animal movements",
                                command=lambda: DLCPredictBehavior(behavior_config_path, BfolderName, saveBFolderName))
predictBehaviourButton.grid(row=12, column=4, padx=2, sticky=N + S + W + E)

if saveFolderName == '' or imgDisplayed == 0:
    predictAllImButton.config(state='disabled')
    generateMaskButton.config(state='disabled')
    predictDLCButton.config(state='disabled')
    saveMatFileCheck.config(state='disabled')
    predictLandmarksButton.config(state='disabled')
    sensoryMapCheck.config(state='disabled')
    predictBehaviourButton.config(state='disabled')


def gui(root):
    root.mainloop()


if __name__ == "__main__":
    gui(root)
