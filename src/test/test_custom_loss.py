import unittest
import numpy as np
import tensorflow as tf
from src.algo.train_semi_supervised import custom_unsupervised_loss

#######################################
# Redéfinition de la fonction de loss #
#######################################
BATCH_SIZE = 4


#####################################
# Définition de notre batch de test #
#####################################


def get_data_rotation(y_true, y_pred):
    y_true_rotation = y_true.copy()
    y_true_rotation[3, 1] = 1
    y_true_rotation[3, 5:] = np.ones(5)
    y_pred_rotation = y_pred.copy()
    angle = y_true_rotation[3, 4]
    for i in range(0, 10, 2):
        y_pred_rotation[3, i] = np.cos(angle) * y_pred_rotation[2, i] - np.sin(angle) * y_pred_rotation[2, i + 1]
        y_pred_rotation[3, i + 1] = np.sin(angle) * y_pred_rotation[2, i] + np.cos(angle) * y_pred_rotation[2, i + 1]
    return y_true_rotation, y_pred_rotation


def get_data_translation(y_true, y_pred):
    y_true_translation = y_true.copy()
    y_true_translation[3, 1] = 0
    y_true_translation[3, 5:] = np.ones(5)
    y_pred_translation = y_pred.copy()
    y_pred_translation[3][::2] = y_pred[2][::2] - np.ones(5) * y_true[3, 2]
    y_pred_translation[3, 1:][::2] = y_pred[2, 1:][::2] - np.ones(5) * y_true[3, 3]
    return y_true_translation, y_pred_translation


def get_data_stack(y_true, y_pred):
    y_true_stack, y_pred_stack = get_data_translation(y_true, y_pred)
    y_pred_stack[2] = y_pred_stack[3]
    y_true_stack, y_pred_stack = get_data_rotation(y_true_stack, y_pred_stack)
    y_pred_stack[2] = y_pred[2]
    y_true_stack[3, 1] = 2
    return y_true_stack, y_pred_stack


y_true = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
                   [-2, 0, -5, 4, 25, 1, 0, 0, 0, 0]], dtype=np.float)
y_pred = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype=np.float)
y_true_translation, y_pred_translation = get_data_translation(y_true, y_pred)
y_true_rotation, y_pred_rotation = get_data_rotation(y_true, y_pred)
y_true_stack, y_pred_stack = get_data_stack(y_true, y_pred)
y_true_multiple = np.vstack((y_true[:2], y_true_translation[2:], y_true_rotation[3]))
y_pred_multiple = np.vstack((y_pred[:2], y_pred_translation[2:], y_pred_rotation[3]))
# y_true_mask_outbox, y_pred_mask_outbox = get_data_translation(y_true, y_pred)
# y_true_mask_outbox[3,6:] = np.zeros(4)

class TestStringMethods(unittest.TestCase):

    def test_supervised_loss(self):
        self.assertAlmostEqual(0, test_keras(y_true[:2], y_pred[:2], nb_transfo=0), delta=10 ** -5)
        self.assertAlmostEqual(test_keras(y_true[:2], y_pred[:2], nb_transfo=0),
                                test_numpy(y_true[:2], y_pred[:2], nb_transfo=0),
                                delta=10 ** -5,
                                msg="Erreur loss supervised")

    def test_unsupervised_translation(self):
        self.assertAlmostEqual(0, test_keras(y_true_translation[2:], y_pred_translation[2:]), delta=10 ** -5)
        self.assertAlmostEqual(test_keras(y_true_translation[2:], y_pred_translation[2:]),
                                test_numpy(y_true_translation[2:], y_pred_translation[2:]),
                                delta=10 ** -5,
                                msg="Erreur loss unsupervised translation")

    def test_unsupervised_rotation(self):
        self.assertAlmostEqual(0, test_keras(y_true_rotation[2:], y_pred_rotation[2:]), delta=10 ** -5)
        self.assertAlmostEqual(test_keras(y_true_rotation[2:], y_pred_rotation[2:]),
                                           test_numpy(y_true_rotation[2:], y_pred_rotation[2:]),
                                           delta=10 ** -5,
                                           msg="Erreur loss unsupervised rotation")

    def test_unsupervised_stack(self):
        self.assertAlmostEqual(0, test_keras(y_true_stack[2:], y_pred_stack[2:]), delta=10 ** -5)
        self.assertAlmostEqual(test_keras(y_true_stack[2:], y_pred_stack[2:]),
                               test_numpy(y_true_stack[2:], y_pred_stack[2:]),
                               delta=10 ** -5,
                               msg="Erreur loss unsupervised stack")

    def test_unsupervised_multiple_transformation(self):
        self.assertAlmostEqual(0, test_keras(y_true_multiple, y_pred_multiple, nb_transfo=2), delta=10 ** -5)
        self.assertAlmostEqual(test_keras(y_true_multiple, y_pred_multiple, nb_transfo=2),
                               test_numpy(y_true_multiple, y_pred_multiple, nb_transfo=2),
                               delta=10 ** -5,
                               msg="Erreur loss unsupervised multiple")

    def test_fp_out_of_box(self):
        def get_result():
            return np.sum(np.square(y_pred[2,:2] - np.array((y_true[3,2]+y_pred[3,0], y_true[3,3]+y_pred[3,1]))))/(2.0+10**-5)
        self.assertAlmostEqual(get_result(), test_keras(y_true, y_pred), delta=10**-5)


if __name__ == '__main__':
    unittest.main()


# Execution du calcul de la loss avec notre fonction
def test_keras(y_true, y_pred, nb_transfo=1):
    with tf.Session() as default:
        result = custom_unsupervised_loss(tf.constant(y_true, dtype='float32'), tf.constant(y_pred, dtype='float32'),
                                          nb_transfo=nb_transfo)
        return result.eval()


# Execution du calcul de la loss avec numpy
def test_numpy(y_true, y_pred, nb_transfo=1):
    def get_loss_supervised(y_true, y_pred):
        y_true_l = y_true[np.where(y_true[:, 0] >= -1)]
        y_pred_l = y_pred[np.where(y_true[:, 0] >= -1)]
        if y_pred_l.shape[0] == 0:
            return 0
        else:
            return np.mean(np.square(y_pred_l - y_true_l))

    def get_loss_unsupervised(y_true, y_pred):
        bs, nb_fp = y_true.shape
        y_true_ul = y_true[np.where(y_true[:, 0] < -1)].reshape(-1, (nb_transfo + 1) * nb_fp)
        y_pred_ul = y_pred[np.where(y_true[:, 0] < -1)].reshape(-1, (nb_transfo + 1) * nb_fp)
        loss_ul = 0
        for i in range(1, nb_transfo + 1):
            transfo_loss = get_backward_transfo(y_true_ul, y_pred_ul, i)
            loss_ul += transfo_loss
        return loss_ul  # np.mean(np.square(y_pred_ul[:, :10] - y_transfo[:, 10:]))

    def get_backward_transfo(y_true, y_pred, indice_nombre_transfo):
        id_transfo = y_true[0, indice_nombre_transfo * 10 + 2 - 1]
        if id_transfo == 0:
            y_transform = get_backward_translation(y_true, y_pred, indice_nombre_transfo)
        elif id_transfo == 1:
            y_transform = get_backward_rotation(y_true, y_pred, indice_nombre_transfo)
        elif id_transfo == 2:
            y_transform = get_backward_stack(y_true, y_pred, indice_nombre_transfo)
        else:
            raise ValueError("transformation id is wrong in test calculation:" + str(id_transfo))
        return np.mean(
            np.power(y_pred[:, :10] - y_transform[:, indice_nombre_transfo * 10:indice_nombre_transfo * 10 + 10], 2))

    def get_backward_translation(y_true, y_pred, indice_nombre_transfo):
        val_translate_x = y_true[:, indice_nombre_transfo * 10 + 2].copy()
        val_translate_y = y_true[:, indice_nombre_transfo * 10 + 3].copy()
        transfo = np.zeros(y_pred.shape)
        transfo[:, indice_nombre_transfo * 10:indice_nombre_transfo * 10 + 10:2] = np.ones(
            (y_pred.shape[0], 5)) * val_translate_x
        transfo[:, indice_nombre_transfo * 10 + 1:indice_nombre_transfo * 10 + 10:2] = np.ones(
            (y_pred.shape[0], 5)) * val_translate_y
        return y_pred + transfo

    def get_backward_rotation(y_true, y_pred, indice_nombre_transfo):
        """
        X = cos(- angle) * x - sin(- angle) * y
        Y = sin(- angle) * x + cos(- angle) * y
        """
        angle = y_true[:, indice_nombre_transfo * 10 + 4]
        y_transfo = y_pred.copy()
        x = y_pred[:, indice_nombre_transfo * 10:indice_nombre_transfo * 10 + 10:2].copy()
        y = y_pred[:, indice_nombre_transfo * 10 + 1:indice_nombre_transfo * 10 + 10:2].copy()
        y_transfo[:, indice_nombre_transfo * 10:indice_nombre_transfo * 10 + 10:2] = np.cos(-angle) * x - np.sin(
            -angle) * y
        y_transfo[:, indice_nombre_transfo * 10 + 1:indice_nombre_transfo * 10 + 10:2] = np.sin(-angle) * x + np.cos(
            -angle) * y
        return y_transfo

    def get_backward_stack(y_true, y_pred, indice_nombre_transfo):
        y_transfo = get_backward_rotation(y_true, y_pred, indice_nombre_transfo)
        return get_backward_translation(y_true, y_transfo, indice_nombre_transfo)

    result_l = get_loss_supervised(y_true, y_pred)
    if nb_transfo > 0:
        result_ul = get_loss_unsupervised(y_true, y_pred)
    else:
        result_ul = 0
    return result_l + result_ul
