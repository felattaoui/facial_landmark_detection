import numpy as np
from tensorflow.keras.models import load_model
import src.utils.config as config
import src.utils.plot_points_on_faces as plot_points




def load_my_model(path):
    model = load_model(path)
    return model


def predict_and_compute_losses(model, generator):
    prediction = model.predict(generator, steps=None, max_queue_size=10, workers=config.cfg.TRAIN_PARAM.WORKERS,
                               use_multiprocessing=False,
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

