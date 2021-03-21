import numpy as np
from imgaug import augmenters as iaa
from tensorflow.keras.models import load_model
import src.utils.utils as fp_utils
import src.algo.batch_generator_supervised as sup_batch

import src.utils.config as config

import src.utils.plot_points_on_faces as plot_points
import time

test_txt_file = 'C:\\Users\\MS_BGD\\PycharmProjects\\facial_landmark_detection\\data\\valid_5000_5FP.txt'
image_dir = r'../../data/'
TRAINING_SIZE = 96
MIN_OBJECT_SIZE = 0
NUM_PARAMETERS = 2 * 5
BATCH_SIZE = 64
USE_MULTIPROCESSING = False
WORKERS = 8
CLASSES = ('face')


test_data_list = fp_utils.load_txt_5FP_and_box(config.cfg.TRAIN_PATH.VAL_TXT_FILE, config.cfg.TRAIN_PATH.IMAGE_DIR)

# Prediction Generator
print("predict generator")
img_aug_conf_valid = iaa.Sequential([fp_utils.RedefineBoxes()], random_order=False)

img_aug_conf_valid_unsup = iaa.Sequential([fp_utils.RedefineBoxes()], random_order=False)


print("test generator")
val_gen = sup_batch.BatchGeneratorSupervised(batch_size=config.cfg.TRAIN_PARAM.BATCH_SIZE,
                                             training_size=config.cfg.TRAIN_PARAM.TRAINING_SIZE,
                                             data_list=test_data_list,
                                             classes=CLASSES,
                                             shuffle=True,
                                             image_normalization_fn=fp_utils.normalize_image,
                                             label_normalization_fn=fp_utils.normalize_label,
                                             img_aug_conf=img_aug_conf_valid,
                                             encoding_fn=fp_utils.encode_5FP
                                             )


def load_my_model(path = "C:\\Users\\MS_BGD\\PycharmProjects\\facial_landmark_detection\\models\\supervised\\new"
                         "\\train_supervised_2021_03_21_02_10_19_10000_valid_5000.keras.model"):
    model = load_model(path)
    return model


def predict_and_compute_losses(model, generator):
    prediction = model.predict(generator, steps=None, max_queue_size=10, workers=WORKERS,
                               use_multiprocessing=USE_MULTIPROCESSING,
                               verbose=1)

    print('\n Compute home made losses...')
    mse, eyes_nose_lips_mse = compute_losses(generator, prediction)
    return prediction, mse, eyes_nose_lips_mse


def compute_losses(gen, prediction):
    eyes_nose_lips_mse = np.zeros(5)
    first_idx = 0
    mse = 0.
    for batch in gen:
        label_batch = batch[1]

        batch_size = label_batch.shape[0]
        last_idx = batch_size + first_idx
        pred_batch = prediction[first_idx:last_idx, :]

        squareDiff = np.power(label_batch - pred_batch, 2)
        mse += np.sum(squareDiff)

        for i in range(0, 10, 2):
            eyes_nose_lips_mse[i // 2] += np.sum((squareDiff[:, i] + squareDiff[:, i + 1]))

        first_idx = last_idx

    mse = mse / (prediction.shape[0] * prediction.shape[1])
    eyes_nose_lips_mse = eyes_nose_lips_mse / (prediction.shape[0] * 2)

    return mse, eyes_nose_lips_mse


def compute_results(sizeOfTrain, gen, supervised=True, plot=False, nbImagestoPlot=0):
    model = load_my_model(sizeOfTrain, supervised)
    # result = model.evaluate(gen)
    # print(dict(zip(model.metrics_names, result)))
    prediction, mse, eyes_nose_lips_mse = predict_and_compute_losses(model, gen)
    print('\n MSE globale home made', mse)

    if plot and supervised:
        plot_points.plot_from_generator(gen, nbImagestoPlot, prediction, supervised)

    if plot and supervised == False:
        plot_points.plot_from_generator(gen, nbImagestoPlot, prediction, False)

    points = ['left eye', 'right eye', 'nose', 'left lips corner', 'right lips corner']
    for j in range(5):
        print('\n MSE for ' + points[j] + ' for a training with %d images\n' % sizeOfTrain, eyes_nose_lips_mse[j])
    return mse, eyes_nose_lips_mse


# Main
if __name__ == '__main__':

    a = time.time()

    my_model = load_my_model()
    prediction, mse, eyes_nose_lips_mse = predict_and_compute_losses(my_model, val_gen)
    plot_points.plot_from_generator(val_gen, 20, prediction, True)


    print('Temps d execution en secondes :', time.time() - a)
