import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import src.utils.config as config

TRAINING_SIZE = 96
image_dir = config.cfg.TRAIN_PATH.IMAGE_DIR


def from_file_to_matrices(txt_file, image_dir, nb_images):
    raw = list(np.loadtxt(txt_file, dtype=bytes).astype(str))

    first_img = cv2.imread(image_dir + raw[0][0])  # in order to retrieve resolution and channel of the data
    height, width, channels = first_img.shape

    images = np.zeros((nb_images, height, width, channels))
    coord = np.zeros((nb_images, 10))

    for i, line in enumerate(raw):
        path_to_image = line[0]
        img = cv2.imread(image_dir + path_to_image)
        images[i] = img
        coordinates = line[2:12]
        coordinates = coordinates.astype(np.float)
        coord[i] = coordinates
        if i == nb_images - 1:
            break
    batch = (images, coord)
    return batch


def from_BGR_to_RGB(matrix_image):
    if matrix_image.shape[2] != 3: return
    b = matrix_image[:, :, 0]
    g = matrix_image[:, :, 1]
    r = matrix_image[:, :, 2]
    rg = np.dstack((r, g))
    rgb = np.dstack((rg, b))
    return rgb


def denorm_image(matrix_image):
    matrix_image = matrix_image * 255
    return matrix_image


def denorm_image_mv2(matrix_image):
    matrix_image = matrix_image + 1
    matrix_image = matrix_image * 127.5
    matrix_image = (np.ceil(matrix_image)).astype(int)
    return matrix_image


def denorm_labels(labels):
    c = np.array(10 * [1 / float(TRAINING_SIZE)])
    return labels / c


def plot_from_batch(batch, nb_images, denorm, pred=None, unsupervised=False, mv2=False):
    for i in range(min(batch[0].shape[0], nb_images)):
        image = batch[0][i]
        label = batch[1][i]
        predict = None
        if pred is not None: predict = pred[i, :]
        if denorm == True:
            if mv2: image = denorm_image_mv2(image)
            else: image = denorm_image(image)
            label = denorm_labels(label)
            if pred is not None: predict = denorm_labels(predict)
        image = image.astype(np.uint8)
        image = from_BGR_to_RGB(image)
        if unsupervised:
            plot_points_on_face(image)
        else:
            plot_points_on_face(image, label, predict)
        picture = Image.fromarray(image)
        picture.show()
    return min(batch[0].shape[0], nb_images)


def plot_from_generator(generator, nb_images, prediction=None, unsupervised=False, mv2=False):
    cp = 0
    nbim = nb_images
    last_idx = 0
    for gen in generator:
        batch_size = min(gen[1].shape[0], nbim) + last_idx
        if prediction is None:
            nbImagesGenerated = plot_from_batch(gen, nbim, True, unsupervised=unsupervised, mv2=mv2)
        else:
            nbImagesGenerated = plot_from_batch(gen, nbim, True, prediction[last_idx:batch_size, :], mv2=mv2)

        cp = cp + nbImagesGenerated
        nbim = nbim - nbImagesGenerated
        if cp >= nb_images: return
        last_idx += batch_size


def plot_points_on_face(matrix_image, labels=None, pred=None):
    # define label color : colors = [red, green, blue, yellow, pink]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    if labels is not None:  # and labels.shape[1] == 10:
        for i in range(0, 10, 2):
            (x, y) = (int(float(labels[i])), int(float(labels[i + 1])))
            cv2.circle(matrix_image, (x, y), 1, colors[i // 2], thickness=-1)

    if pred is not None:  # and pred.shape[1]==10:
        for j in range(0, 10, 2):
            (x1, y1) = (int(float(pred[j])), int(float(pred[j + 1])))
            cv2.drawMarker(matrix_image, (x1, y1), colors[j // 2], markerType=cv2.MARKER_CROSS, markerSize=2,
                           thickness=1, line_type=cv2.LINE_AA)

    plt.imshow(matrix_image)
    plt.show()
