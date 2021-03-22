import numpy as np
from imgaug import augmenters as iaa
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.losses import logcosh, mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import time
import src.utils.utils as utils
import src.algo.batch_generator_supervised as sup_batch
import src.utils.config as config

# import tensorflow as tf
# gpu = tf.config.experimental.list_physical_devices('GPU')
# if gpu:
#     # Restreindre la consommation GPU de Tensorflow qui dépasse la capacité standard du GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpu[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2700)])
#         logical_gpu = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpu), "Physical GPUs,", len(logical_gpu), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


def train(workers):
    classes = ('face')
    img_aug_conf = iaa.Sequential([
        iaa.Affine(rotate=(-35, 35)),
        # iaa.PerspectiveTransform(scale=(0, 0.10)),
        # iaa.CropAndPad(percent=(-0.0, -0.15)),
        iaa.Sometimes(0.5, utils.Fliplr_5fp()),
        # iaa.Add((-20, +20), per_channel=True),
        # iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.1))),
        # iaa.Sometimes(0.5, iaa.Grayscale(1.0)),
        # iaa.Sharpen((0.0, 0.5)),
        utils.RedefineBoxes()
    ],
        random_order=False
    )
    img_aug_conf_valid = iaa.Sequential([
        utils.RedefineBoxes()
    ],
        random_order=False
    )

    train_data_list = utils.load_txt_5FP_and_box(config.cfg.TRAIN_PATH.TRAIN_TXT_FILE, config.cfg.TRAIN_PATH.IMAGE_DIR)

    val_data_list = utils.load_txt_5FP_and_box(config.cfg.TRAIN_PATH.VAL_TXT_FILE, config.cfg.TRAIN_PATH.IMAGE_DIR)

    print("training generator")
    train_gen = sup_batch.BatchGeneratorSupervised(batch_size=config.cfg.TRAIN_PARAM.BATCH_SIZE,
                                                   training_size=config.cfg.TRAIN_PARAM.TRAINING_SIZE,
                                                   data_list=train_data_list,
                                                   classes=classes,
                                                   shuffle=True,
                                                   image_normalization_fn=utils.normalize_image,
                                                   label_normalization_fn=utils.normalize_label,
                                                   img_aug_conf=img_aug_conf,
                                                   encoding_fn=utils.encode_5FP
                                                   )
    print("validation generator")
    val_gen = sup_batch.BatchGeneratorSupervised(batch_size=config.cfg.TRAIN_PARAM.BATCH_SIZE,
                                                 training_size=config.cfg.TRAIN_PARAM.TRAINING_SIZE,
                                                 data_list=val_data_list,
                                                 classes=classes,
                                                 shuffle=True,
                                                 image_normalization_fn=utils.normalize_image,
                                                 label_normalization_fn=utils.normalize_label,
                                                 img_aug_conf=img_aug_conf_valid,
                                                 encoding_fn=utils.encode_5FP
                                                 )

    n_batches_train, _ = divmod(np.shape(train_data_list)[
                                0], config.cfg.TRAIN_PARAM.BATCH_SIZE)
    n_batches_eval, _ = divmod(np.shape(val_data_list)[
                               0], config.cfg.TRAIN_PARAM.BATCH_SIZE)

    # build the model
    if config.cfg.TRAIN_PATH.PATH_MODEL is None:
        my_model = utils.mobilenet_v2_custom(config.cfg.TRAIN_PARAM.NUM_PARAMETERS)
        #my_model = utils.vgg19_custom(config.cfg.TRAIN_PARAM.NUM_PARAMETERS)
    else:
        my_model = load_model(config.cfg.TRAIN_PATH.PATH_MODEL)

    my_model.summary()

    # callbacks

    cbs = [
        callbacks.ModelCheckpoint(config.cfg.TRAIN_PATH.KERAS_MODEL, monitor='mean_squared_error', verbose=1,
                                  save_best_only=True, save_weights_only=False, mode='min', period=1),

        callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0, patience=5, verbose=1, mode='min'),

        callbacks.TensorBoard(log_dir=config.cfg.TRAIN_PATH.TENSORBOARD_LOGS, histogram_freq=0,
                              batch_size=config.cfg.TRAIN_PARAM.BATCH_SIZE,  write_graph=True, write_grads=False,
                              write_images=False),

        callbacks.ReduceLROnPlateau(monitor='mean_squared_error', factor=0.5, patience=5, verbose=1, mode='min',
                                    cooldown=0,min_lr=9e-7)
    ]


    # fit
    if config.cfg.TRAIN_PATH.PATH_MODEL is None:
        print("model_compile")
        my_model.compile(loss=mean_squared_error, optimizer=Adam(lr=1e-4), metrics=[mean_squared_error])
        print("model_generator")
        my_model.fit_generator(
            generator=train_gen,
            validation_data=val_gen,
            steps_per_epoch=n_batches_train,
            validation_steps=n_batches_eval,
            epochs=config.cfg.TRAIN_PARAM.NB_EPOCH,
            max_queue_size=40,
            workers=workers,
            use_multiprocessing=config.cfg.TRAIN_PARAM.USE_MULTIPROCESSING,
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
            epochs=config.cfg.TRAIN_PARAM.NB_EPOCH,
            max_queue_size=40,
            workers=workers,
            callbacks=cbs,
            shuffle=True,
            verbose=1
        )


if __name__ == '__main__':
    start_time = time.time()
    train(config.cfg.TRAIN_PARAM.WORKERS)
    # Affichage du temps d execution
    print("Temps d execution : %s secondes ---" % (time.time() - start_time))