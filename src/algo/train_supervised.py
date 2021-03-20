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

forFarid = False
if forFarid:
    import tensorflow as tf
    import keras
    print('Fix for new Nvidia GPU computation')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.set_session(sess)


def train(workers):
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
    img_aug_conf_valid = iaa.Sequential([
        utils.RedefineBoxes()
    ],
        random_order=False
    )

    train_data_list = utils.load_txt_5FP_and_box(
        config.cfg.TRAIN_PATH.TRAIN_TXT_FILE, config.cfg.TRAIN_PATH.IMAGE_DIR)
    val_data_list = utils.load_txt_5FP_and_box(
        config.cfg.TRAIN_PATH.VAL_TXT_FILE, config.cfg.TRAIN_PATH.IMAGE_DIR)

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
        input_data = Input(name='input_data', shape=(
            config.cfg.TRAIN_PARAM.TRAINING_SIZE, config.cfg.TRAIN_PARAM.TRAINING_SIZE, 3), dtype='float32')
        '''y_pred = utils.graph_dense_fp(
            input_data, config.cfg.TRAIN_PARAM.TRAINING_SIZE, config.cfg.TRAIN_PARAM.NUM_PARAMETERS)'''

        # my_model = Model(inputs=input_data, outputs=y_pred)
        my_model = utils.mobilenet_custom(10)
    else:
        my_model = load_model(config.cfg.TRAIN_PATH.PATH_MODEL)

    my_model.summary()

    # fit
    if config.cfg.TRAIN_PATH.PATH_MODEL is None:
        print("model_compile")
        my_model.compile(loss=mean_squared_error, optimizer=Adam(
            lr=1e-4), metrics=[mean_squared_error])
        print("model_generator")
        my_model.fit_generator(
            generator=train_gen,
            validation_data=val_gen,
            steps_per_epoch=n_batches_train,
            validation_steps=n_batches_eval,
            epochs=200,
            max_queue_size=40,
            workers=workers,
            use_multiprocessing=True,
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
            epochs=500,
            max_queue_size=40,
            workers=workers,
            shuffle=True,
            verbose=1
        )


if __name__ == '__main__':
    start_time = time.time()
    train(config.cfg.TRAIN_PARAM.WORKERS)
    # Affichage du temps d execution
    print("Temps d execution : %s secondes ---" % (time.time() - start_time))