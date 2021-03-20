import numpy as np
from imgaug import augmenters as iaa
from tensorflow.keras.models import load_model
import src.utils.utils as fp_utils
import src.algo.train_supervised as supervised
import src.algo.train_semi_supervised as unsupervised

import matplotlib.pyplot as plt

# from keras.losses import mean_squared_error

import src.utils.plot_points_on_faces as plot_points
import time

forFarid = True

if forFarid:
    import tensorflow as tf
    import keras
    print('Fix for new Nvidia GPU computation')
    core_config=tf.ConfigProto()
    core_config.gpu_options.allow_growth = True
    session = tf.Session(config=core_config)
    keras.backend.set_session(session)


test_txt_file = r'../../data/valid_5000_5FP.txt'
image_dir = r'../../data/'
TRAINING_SIZE = 96
MIN_OBJECT_SIZE = 0
NUM_PARAMETERS = 2 * 5
BATCH_SIZE = 64
USE_MULTIPROCESSING = False
WORKERS = 8
CLASSES = ('face')


# Load test data
test_data_list = fp_utils.load_txt_5FP_and_box(test_txt_file, image_dir)

# Prediction Generator
print("predict generator")
img_aug_conf_valid = iaa.Sequential([fp_utils.RedefineBoxes()],random_order=False)

img_aug_conf_valid_unsup = iaa.Sequential([fp_utils.RedefineBoxes()],random_order=False)


val_gen = supervised.BatchGenerator(batch_size=BATCH_SIZE,
                                    data_list=test_data_list,
                                    classes=CLASSES,
                                    shuffle=False,
                                    image_normalization_fn=fp_utils.normalize_image,
                                    label_normalization_fn=fp_utils.normalize_label,
                                    img_aug_conf=img_aug_conf_valid,
                                    encoding_fn=fp_utils.encode_5FP
                                    )



def load_my_model(sizeOfTrain, supervised = True):
    if sizeOfTrain not in [100, 1000, 10000, 100000, 1000000]:
        print('loading file failed')
        return

    if supervised :
        print('Loading supervised train_%d model' %sizeOfTrain)
        model_path = "../../models/supervised/supervised_train_%d_valid_5000/model_%d_valid_5000.keras.model" % (sizeOfTrain, sizeOfTrain)
        model = load_model(model_path)
    else :
        print('Loading unsupervised train_%d model' %sizeOfTrain)
        model_path = "../../models/unsupervised/translation et rotation/train_%d/model_unsup_train_%d_valid_5000.keras.model" % (sizeOfTrain, sizeOfTrain)
        model = load_model(model_path, custom_objects={'custom_unsupervised_loss': unsupervised.custom_unsupervised_loss})

    return model


def predict_and_compute_losses(model, generator):
    prediction = model.predict(generator, steps=None, max_queue_size=10, workers=WORKERS, use_multiprocessing=USE_MULTIPROCESSING,
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
            eyes_nose_lips_mse[i//2] += np.sum((squareDiff[:,i] + squareDiff[:,i+1]))

        first_idx = last_idx

    mse = mse / (prediction.shape[0] * prediction.shape[1])
    eyes_nose_lips_mse = eyes_nose_lips_mse / (prediction.shape[0] * 2)

    return mse, eyes_nose_lips_mse


def compute_results(sizeOfTrain, gen, supervised=True, plot=False, nbImagestoPlot = 0):
    model = load_my_model(sizeOfTrain, supervised)
    #result = model.evaluate(gen)
    #print(dict(zip(model.metrics_names, result)))
    prediction, mse, eyes_nose_lips_mse = predict_and_compute_losses(model, gen)
    print('\n MSE globale home made', mse)

    if plot and supervised:
        plot_points.plot_from_generator(gen,nbImagestoPlot,prediction,supervised)


    if plot and supervised==False:
        plot_points.plot_from_generator(gen, nbImagestoPlot, prediction, False)

    points = ['left eye', 'right eye', 'nose', 'left lips corner', 'right lips corner']
    for j in range(5):
        print('\n MSE for ' + points[j] + ' for a training with %d images\n' % sizeOfTrain, eyes_nose_lips_mse[j])
    return mse, eyes_nose_lips_mse



# Main
if __name__ == '__main__':

    a=time.time()

    MSE_sup = np.zeros(5)
    MSE_EyesNoseLips_sup = np.zeros([5,5])

    MSE_unsup = np.zeros(5)
    MSE_EyesNoseLips_unsup = np.zeros([5,5])

    plot_sup = True
    supervised = True

    plot_unsup = False
    semi_supervised = False
    for i, size_of_train in enumerate([100, 1000, 10000, 100000]):

        if supervised:
            mse, lossENL = compute_results(size_of_train, val_gen, supervised, plot_sup, 5)
            MSE_sup[i] = mse
            MSE_EyesNoseLips_sup[i:] = lossENL

        if semi_supervised :
            mse_unsup, lossENL_unsup = compute_results(size_of_train, val_gen, False, plot_unsup, 5)#val_gen_unsup
            MSE_unsup[i] = mse_unsup
            MSE_EyesNoseLips_unsup[i:] = lossENL_unsup

    save = False
    if save:
        if supervised:
            np.savetxt('MSE_sup.csv', MSE_sup, fmt='%1.4e', delimiter=",")
            np.savetxt('MSE_EyesNoseLips_sup.csv', MSE_EyesNoseLips_sup, fmt='%1.4e', delimiter=",")
        if semi_supervised :
            np.savetxt('MSE_unsup.csv', mse_unsup, fmt='%1.4e', delimiter=",")
            np.savetxt('MSE_EyesNoseLips_unsup.csv', MSE_EyesNoseLips_unsup, fmt='%1.4e', delimiter=",")

    print('Temps d execution en secondes :',time.time()-a)
