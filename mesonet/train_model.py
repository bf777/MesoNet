from mesonet.model import *
from mesonet.data import *
import numpy as np
from keras.callbacks import ModelCheckpoint
from mesonet.utils import parse_yaml


def trainModel(input_file, model_name, log_folder):
    data_gen_args = dict(rotation_range=0.3, width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
                         zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')
    myGene = trainGenerator(2, input_file, 'image', 'label', data_gen_args, save_to_dir=None)
    model = unet()
    model_checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
    history_callback = model.fit_generator(myGene, steps_per_epoch=300, epochs=60, callbacks=[model_checkpoint])
    loss_history = history_callback.history["loss"]
    acc_history = history_callback.history["acc"]
    np_loss_hist = np.array(loss_history)
    np_acc_hist = np.array(acc_history)
    np.savetxt(os.path.join(log_folder, "loss_history.csv"), np_loss_hist, delimiter=",")
    np.savetxt(os.path.join(log_folder, "acc_history.csv"), np_acc_hist, delimiter=",")
    model.save(os.path.join('models', model_name))


def train_model(config_file):
    cfg = parse_yaml(config_file)
    input_file = cfg['input_file']
    model_name = cfg['model_name']
    log_folder = cfg['log_folder']
    trainModel(input_file, model_name, log_folder)
