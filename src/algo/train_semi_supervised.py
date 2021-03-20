import numpy as np
import os
import imgaug.augmenters as iaa
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras import backend as K
import time
import src.utils.utils as utils
import src.algo.batch_generator_semi_supervised as semi_sup_batch
import src.algo.batch_generator_supervised as sup_batch
import tensorflow as tf

forFarid = False
if forFarid:
    import keras
    print('Fix for new Nvidia GPU computation')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.set_session(sess)

size_of_train = 100
# Path of the model to continue training if new training put None
path_model = None

if size_of_train // 100 not in [1, 10, 100, 1000, 10000]:
    print('Erreur de chargement de fichier de train')

train_txt_file = r'../../data/train_%d_5FP.txt' % size_of_train
val_txt_file = r'../../data/valid_5000_5FP.txt'
image_dir = r'../../data/'

TRAINING_SIZE = 96
MIN_OBJECT_SIZE = 0
NUM_PARAMETERS = 2 * 5
BATCH_SIZE = 64
# MODEL_NAME = model_name(r'../../Models_dense')
# MODEL_NAME = utils.model_name(size_of_train, r'../../models/unsupervised/new/train_%d' % size_of_train)
MODEL_NAME = utils.model_name(size_of_train,
                              r'C:\Users\MS_BGD\PycharmProjects\semi_supervised_CNN\models\unsupervised\new')

os.makedirs(MODEL_NAME, exist_ok=True)
KERAS_MODEL = MODEL_NAME + '.keras.model'
TENSORBOARD_LOGS = MODEL_NAME + '.logs'
USE_MULTIPROCESSING = True
WORKERS = 6
TRANSFO_TO_APPLY = ['translation']
NUM_TRANSFO = len(TRANSFO_TO_APPLY)
NB_IMG_LABEL = int(size_of_train / 10)
# Hyper parameters
UNSUPERVISED_LOSS_REG = 1
BATCH_SIZE_LABELIZE = int(BATCH_SIZE / 4)
BATCH_SIZE_UNLABELIZE = int((BATCH_SIZE - BATCH_SIZE_LABELIZE) / (NUM_TRANSFO + 1))


def _label_to_points(label, image_shape):
    # be careful, only 5FP are output
    return [[float(TRAINING_SIZE) * label[2 * i], float(TRAINING_SIZE) * label[2 * i + 1]] for i in range(5)]


def custom_unsupervised_loss(y_true, y_pred, nb_transfo=NUM_TRANSFO):
    nb_fp_point = 10

    def get_supervised_loss(y_true, y_pred):
        y_pred_l = tf.gather(y_pred, tf.where(tf.greater_equal(y_true[:, 0], -1)), axis=0)
        y_true_l = tf.gather(y_true, tf.where(tf.greater_equal(y_true[:, 0], -1)), axis=0)
        return tf.cond(tf.equal(tf.size(y_pred_l), 0),
                       lambda: tf.constant([0.0]),
                       lambda: tf.reduce_mean(tf.square(y_true_l - y_pred_l)))

    def get_unsupervised_loss(y_true, y_pred):
        y_pred_ul = tf.gather(y_pred, tf.where(tf.less(y_true[:, 0], -1)), axis=0)
        y_true_ul = tf.gather(y_true, tf.where(tf.less(y_true[:, 0], -1)), axis=0)
        # Reshape to align unlabelised image and its transformation
        y_pred_ul = K.reshape(y_pred_ul, (-1, nb_fp_point * (nb_transfo + 1)))
        y_true_ul = K.reshape(y_true_ul, (-1, nb_fp_point * (nb_transfo + 1)))
        return tf.cond(tf.equal(tf.size(y_pred_ul), 0),
                       lambda: tf.constant([0.0]),
                       lambda: compute_loss_ul(y_true_ul, y_pred_ul)[0])

    def compute_loss_ul(y_true_ul, y_pred_ul):
        # Apply backtransformation + calculate loss for each transformation applied
        loss_ul = tf.zeros(shape=1)
        for i in range(nb_transfo):
            id_begin_transfo = (i + 1) * nb_fp_point
            transfo_loss = get_loss_transformation(y_true_ul, y_pred_ul, id_begin_transfo)
            loss_ul = loss_ul + transfo_loss
        return loss_ul / nb_transfo

    def get_loss_transformation(y_true, y_pred, id_begin_transfo):
        """
        Return the loss for one of the transformation computed.
        :param y_true:
        :param y_pred:
        :param id_begin_transfo: indicate the starting position of the transformation information in the tensor
        :return:
        """
        indice_type_transfo = tf.reduce_mean(y_true[:, id_begin_transfo + 1])  # 0 for translation, 1 for rotation
        y_pred_transform = tf.case([(tf.equal(indice_type_transfo, 0),
                                     lambda: get_backward_translation(y_true, y_pred, id_begin_transfo)),
                                    (tf.equal(indice_type_transfo, 1),
                                     lambda: get_backward_rotation(y_true, y_pred, id_begin_transfo)),
                                    (tf.equal(indice_type_transfo, 2),
                                     lambda: get_backward_stack(y_true, y_pred, id_begin_transfo))],
                                   default=lambda: tf.zeros(tf.shape(y_pred)))
        mask_outbox_fp = get_mask_outbox(y_true, id_begin_transfo)
        return tf.reduce_sum(mask_outbox_fp * tf.square(y_pred[:, :nb_fp_point]
                                                        - y_pred_transform[:,
                                                          id_begin_transfo:id_begin_transfo + nb_fp_point])) / (
                           tf.reduce_sum(mask_outbox_fp) + 10 ** -5)
        # return tf.reduce_mean(tf.square(y_pred[:, :nb_fp_point]
        #                                 - y_pred_transform[:, id_begin_transfo:id_begin_transfo + nb_fp_point]))

    def get_backward_translation(y_true, y_pred, id_begin_transfo):
        """
        Calculate the backward translation from the information stored in y_true on the prediction y_pred
        :param y_true:
        :param y_pred:
        :param id_begin_transfo: indicate the starting position of the transformation information in the tensor
        :return: y_pred with backward translation apply
        """
        val_translate = tf.concat([tf.reshape(y_true[:, id_begin_transfo + 2], (1, -1)),
                                   tf.reshape(y_true[:, id_begin_transfo + 3], (1, -1))], axis=0)  # shape (2,bs)
        indices = tf.reshape(tf.range(nb_fp_point) + id_begin_transfo, (-1, 1))  # shape(10, 1)
        updates = tf.tile(val_translate, [int(nb_fp_point / 2), 1])  # shape (10,bs)
        return tf.transpose(tf.tensor_scatter_nd_add(tf.transpose(y_pred), indices=indices, updates=updates))

    def get_backward_rotation(y_true, y_pred, id_begin_transfo):
        """
        The backward rotation transformation to apply to retrieve the prediction in the original space, which is :
            X = cos(- angle) * x - sin(- angle) * y
            Y = sin(- angle) * x + cos(- angle) * y
        Here we apply the transformation for each double of face points
        :param: y_true
        :param y_pred:
        :param id_begin_transfo: indicate the starting position of the transformation information in the tensor
        :return: y_pred with backward rotation apply
        """
        output = tf.transpose(y_pred)
        angle = tf.reshape(tf.gather(y_true, id_begin_transfo + 4, axis=1), (-1, 1))
        for i in range(id_begin_transfo, id_begin_transfo + nb_fp_point, 2):
            x = tf.reshape(y_pred[:, i], (-1, 1))
            y = tf.reshape(y_pred[:, i + 1], (-1, 1))
            indices = tf.constant([[i], [i + 1]])
            updates = tf.transpose(
                tf.concat([tf.cos(-angle) * x - tf.sin(-angle) * y, tf.sin(-angle) * x + tf.cos(-angle) * y],
                          axis=1))
            output = tf.tensor_scatter_nd_update(output, indices=indices, updates=updates)
        return tf.transpose(output)

    def get_backward_stack(y_true, y_pred, id_begin_transfo):
        """
        Apply all backward transformation to y_pred
        :param y_true:
        :param y_pred:
        :param id_begin_transfo: indicate the starting position of the transformation information in the tensor
        :return:
        Since images are created by translation and then rotation we must apply the transformation in the inverse way,
        rotation and then translation
        """
        y_transform = get_backward_rotation(y_true, y_pred, id_begin_transfo)
        return get_backward_translation(y_true, y_transform, id_begin_transfo)

    def get_mask_outbox(y_true, id_begin_transfo):
        """
        Return a 10 value tensor indicating if a face point is out of the image (0) or in the image (1) This information
        is store in y_true from id 4 to 9. we need to duplicate this information for both x and y in order to apply the
        mask
        :param y_true:
        :return:
        """
        mask_outbox_fp = tf.transpose(tf.ones(tf.shape(y_true), dtype=tf.float32)[:, :10])
        for id_mask, id_ytrue in enumerate(range(id_begin_transfo + 5, id_begin_transfo + nb_fp_point)):
            mask_bool = tf.reshape(y_true[:, id_ytrue], (-1, 1))
            indices = tf.constant([[id_mask * 2], [id_mask * 2 + 1]])
            updates = tf.transpose(tf.concat([mask_bool, mask_bool], axis=1))
            mask_outbox_fp = tf.tensor_scatter_nd_update(mask_outbox_fp, indices=indices, updates=updates)
        return tf.transpose(mask_outbox_fp)

    # Supervised Loss
    loss_l = get_supervised_loss(y_true, y_pred)
    # Unsupervised Loss
    loss_ul = get_unsupervised_loss(y_true, y_pred)
    return loss_l + UNSUPERVISED_LOSS_REG * loss_ul


def train(workers):
    batch_size = BATCH_SIZE
    training_size = TRAINING_SIZE
    classes = ('face')
    img_aug_conf = iaa.Sequential([
        iaa.Affine(rotate=(-35, 35)),
        iaa.PerspectiveTransform(scale=(0, 0.10)),
        iaa.CropAndPad(percent=(-0.0, -0.15)),
        iaa.Sometimes(0.5, utils.Fliplr_5fp()),
        iaa.Add((-20, +20), per_channel=True),
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.1))),
        iaa.Sometimes(0.5, iaa.Grayscale(1.0)),
        iaa.Sharpen((0.0, 0.5)),
        utils.RedefineBoxes()
    ],
        random_order=False
    )
    img_aug_conf_unsup = iaa.Sequential([
        iaa.Add((-20, +20), per_channel=True),
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.1))
                      )
        ,
        iaa.Sometimes(0.5, iaa.Grayscale(1.0)),
        iaa.Sharpen((0.0, 0.5)),
        iaa.Sometimes(0.3, iaa.Cutout(nb_iterations=1)),
        utils.RedefineBoxes()
    ],
        random_order=False
    )
    img_aug_conf_valid = iaa.Sequential([
        utils.RedefineBoxes()
    ],
        random_order=False
    )

    train_data_list = utils.load_txt_5FP_and_box(train_txt_file, image_dir)
    val_data_list = utils.load_txt_5FP_and_box(val_txt_file, image_dir)
    print("training generator")
    train_gen = semi_sup_batch.BatchGeneratorSemiSupervised(batch_size=batch_size,
                                                            training_size=training_size,
                                                            data_list=train_data_list,
                                                            classes=classes,
                                                            shuffle=True,
                                                            image_normalization_fn=utils.normalize_image,
                                                            label_normalization_fn=utils.normalize_label,
                                                            list_transfo=TRANSFO_TO_APPLY,
                                                            img_aug_conf=img_aug_conf,
                                                            img_aug_conf_unsupervised=img_aug_conf_unsup,
                                                            encoding_fn=utils.encode_5FP
                                                            )
    print("validation generator")
    val_gen = sup_batch.BatchGeneratorSupervised(batch_size=batch_size,
                                                 training_size=training_size,
                                                 data_list=val_data_list,
                                                 classes=classes,
                                                 shuffle=True,
                                                 image_normalization_fn=utils.normalize_image,
                                                 label_normalization_fn=utils.normalize_label,
                                                 img_aug_conf=img_aug_conf_valid,
                                                 encoding_fn=utils.encode_5FP
                                                 )

    # val_img, val_lbl = utils.get_one_epoch(val_gen)
    n_batches_train, _ = divmod(np.shape(train_data_list)[0], batch_size)
    n_batches_eval, _ = divmod(np.shape(val_data_list)[0], batch_size)

    # del val_gen

    # build the model
    # from keras import backend as K

    if path_model is None:
        input_data = Input(name='input_data', shape=(TRAINING_SIZE, TRAINING_SIZE, 3), dtype='float32')
        # y_pred = graph(input_data)
        y_pred = utils.graph_dense_fp(input_data, TRAINING_SIZE, NUM_PARAMETERS)
        # y_pred = utils.CNN_test(input_data, TRAINING_SIZE, NUM_PARAMETERS)
        my_model = Model(inputs=input_data, outputs=y_pred)
    else:
        my_model = load_model(path_model, custom_objects={'custom_unsupervised_loss': custom_unsupervised_loss})
    my_model.summary()

    # _, _ = src.utils.networks.get_flops(my_model, verbose=True)

    # callbacksScale
    cbs = [
        callbacks.ModelCheckpoint(KERAS_MODEL, monitor='custom_unsupervised_loss', verbose=1, save_best_only=True,
                                  save_weights_only=False, mode='min', period=1),
        callbacks.EarlyStopping(monitor='custom_unsupervised_loss', min_delta=0, patience=10, verbose=1, mode='min'),
        callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS, histogram_freq=0, batch_size=32, write_graph=True,
                              write_grads=False, write_images=False),
        callbacks.ReduceLROnPlateau(monitor='custom_unsupervised_loss', factor=0.5, patience=5, verbose=1, mode='min',
                                    cooldown=0,
                                    min_lr=9e-7)
    ]

    # fit
    if path_model is None:
        print("model_compile")
        my_model.compile(loss=custom_unsupervised_loss, optimizer=Adam(lr=1e-4), metrics=[custom_unsupervised_loss])

        print("model_generator")
        my_model.fit_generator(
            generator=train_gen,
            validation_data=val_gen,
            steps_per_epoch=n_batches_train,
            validation_steps=n_batches_eval,
            epochs=1000,
            max_queue_size=40,
            workers=workers,
            use_multiprocessing=USE_MULTIPROCESSING,
            callbacks=cbs,
            shuffle=True,
            verbose=1
        )
    else:
        print("model_generator")
        my_model.fit_generator(
            generator=train_gen,
            validation_data=val_gen,
            steps_per_epoch=n_batches_train,
            validation_steps=n_batches_eval,
            epochs=1000,
            max_queue_size=40,
            workers=workers,
            # use_multiprocessing=USE_MULTIPROCESSING,
            callbacks=cbs,
            shuffle=True,
            verbose=1
        )


if __name__ == '__main__':
    start_time = time.time()
    train(WORKERS)
    # Affichage du temps d execution
    print("Temps d execution : %s secondes ---" % (time.time() - start_time))
