"""
MesoNet
Authors: Brandon Forys and Dongsheng Xiao, Murphy Lab
https://github.com/bf777/MesoNet
Licensed under the MIT License (see LICENSE for details)
"""
from mesonet.predict_regions import predictRegion
from mesonet.dlc_predict import DLCPredict
from tkinter import *  # Python 3.x
from tkinter import filedialog
from PIL import Image, ImageTk

import os
import fnmatch
import glob

# The main window of the app
root = Tk()
root.resizable(False, False)

cwd = os.getcwd()
folderName = cwd
saveFolderName = ''
threshold = 0.001
config_path = os.path.join(cwd, 'dlc/config.yaml')
landmark_atlas_img = os.path.join(cwd, 'atlases/landmarks_atlas_512_512.png')
sensory_atlas_img = os.path.join(cwd, 'atlases/sensorymap_atlas_512_512.png')
haveMasks = False
# status = 'Please select a folder with brain images at "Input Folder".'

j = -1
delta = 0
imgDisplayed = 0


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
    imageNameLabel.grid(row=2, column=0, columnspan=1)
    imageNumLabel = Label(root, textvariable=imageNumPrep)
    imageNumLabel.grid(row=2, column=2, columnspan=1)


def FindModelInFolder():
    modelSelect = []
    for file in os.listdir(os.path.join(cwd, 'models')):
        if fnmatch.fnmatch(file, "*.hdf5"):
            modelSelect.append(file)

    modelLabel = Label(root, text="If using U-net,\nselect a model to\nanalyze the brain regions:")
    modelListBox = Listbox(root)
    modelLabel.grid(row=3, column=4, padx=2, sticky=S)
    modelListBox.grid(row=4, column=4, sticky=N)
    for item in modelSelect:
        modelListBox.insert(END, item)
    if len(modelSelect) > 0:
        def onselect(evt):
            global model
            w = evt.widget
            selected = int(w.curselection()[0])
            model = modelListBox.get(selected)
            model = os.path.join(os.path.join(cwd, 'models', model))
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


def PredictDLC(config, input_file, output, atlas, landmark_atlas_img, sensory_atlas_img,
               sensory_match, model, num_images, mat_save, threshold, mask_generate, haveMasks):
    global status
    status = "Processing..."
    root.update()
    if mask_generate and not haveMasks:
        predictRegion(input_file, num_images, model, output, mat_save, threshold, True)
    DLCPredict(config, input_file, output, atlas, landmark_atlas_img, sensory_atlas_img, sensory_match, mat_save,
               threshold)
    saveFolderName = output
    global folderName
    if not atlas:
        folderName = os.path.join(saveFolderName, "output_overlay")
    elif atlas:
        folderName = saveFolderName
    status = "Processing complete! RED = poor region match or first image, GREEN = perfect region match, BLUE = close" \
             "region match"
    root.update()
    ImageDisplay(1, folderName, 1)


Title = root.title("MesoNet Analyzer")

canvas = Canvas(root, width=520, height=520)
canvas.grid(row=3, column=0, columnspan=4, rowspan=8, sticky=N + S + W)

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

# Image controls
# Buttons below will only display if an image is displayed
nextButton = Button(root, text="->", command=lambda: ImageDisplay(1, folderName, 0))
nextButton.grid(row=11, column=2, columnspan=1)
previousButton = Button(root, text="<-", command=lambda: ImageDisplay(-1, folderName, 0))
previousButton.grid(row=11, column=0, columnspan=1)

# Bind right and left arrow keys to forward/backward controls
root.bind('<Right>', forward)
root.bind('<Left>', backward)

# Buttons for making predictions
# Buttons below will only be active if a save file has been selected
mat_save = IntVar()
atlas = IntVar()
sensory_align = IntVar()
saveMatFileCheck = Checkbutton(root, text="Save predicted regions\nas .mat files", variable=mat_save)
saveMatFileCheck.grid(row=5, column=4, padx=2, sticky=N+S+W)
sensoryMapCheck = Checkbutton(root, text="Align using sensory map", variable=sensory_align)
sensoryMapCheck.grid(row=6, column=4, padx=2, sticky=N+S+W)
generateMaskButton = Button(root, text="Get boundaries of brain\nusing U-net",
                            command=lambda: PredictRegions(folderName, picLen,
                                                           os.path.join(cwd, 'models/unet_bundary.hdf5'),
                                                           saveFolderName, int(mat_save.get()),
                                                           threshold, True))
generateMaskButton.grid(row=7, column=4, padx=2, sticky=N + S + W + E)
predictLandmarksButton = Button(root, text="Predict landmark locations",
                          command=lambda: PredictDLC(config_path, folderName, saveFolderName, True,
                                                     landmark_atlas_img, sensory_atlas_img,
                                                     int(sensory_align.get()),
                                                     os.path.join(cwd, 'models/unet_bundary.hdf5'), picLen,
                                                     int(mat_save.get()), threshold, False, haveMasks))
predictLandmarksButton.grid(row=8, column=4, padx=2, sticky=N + S + W + E)
predictDLCButton = Button(root, text="Predict brain regions\nusing landmarks",
                          command=lambda: PredictDLC(config_path, folderName, saveFolderName, False,
                                                     landmark_atlas_img, sensory_atlas_img,
                                                     int(sensory_align.get()),
                                                     os.path.join(cwd, 'models/unet_bundary.hdf5'), picLen,
                                                     int(mat_save.get()), threshold, True, haveMasks))
predictDLCButton.grid(row=9, column=4, padx=2, sticky=N + S + W + E)
predictAllImButton = Button(root, text="Predict brain regions directly\nusing pretrained U-net model",
                            command=lambda: PredictRegions(folderName, picLen, model, saveFolderName,
                                                           int(mat_save.get()), threshold, False))
predictAllImButton.grid(row=10, column=4, padx=2, sticky=N + S + W + E)


if saveFolderName == '' or imgDisplayed == 0:
    predictAllImButton.config(state='disabled')
    generateMaskButton.config(state='disabled')
    predictDLCButton.config(state='disabled')
    saveMatFileCheck.config(state='disabled')
    predictLandmarksButton.config(state='disabled')
    sensoryMapCheck.config(state='disabled')
root.mainloop()