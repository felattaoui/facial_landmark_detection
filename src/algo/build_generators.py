import numpy as np
from imgaug import augmenters as iaa
import src.utils.utils as utils
import src.algo.batch_generator as sup_batch
import conf as config


def build_generators(data_aug=True):
    img_aug_conf_valid = iaa.Sequential([
        utils.RedefineBoxes()
    ],
        random_order=False
    )
    if data_aug:
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
    else:
        img_aug_conf = img_aug_conf_valid

    train_data_list = utils.load_txt_5FP_and_box(config.cfg.TRAIN_PATH.TRAIN_TXT_FILE, config.cfg.TRAIN_PATH.IMAGE_DIR)

    val_data_list = utils.load_txt_5FP_and_box(config.cfg.TRAIN_PATH.VAL_TXT_FILE, config.cfg.TRAIN_PATH.IMAGE_DIR)
    classes = ('faces')


    print("training generator")
    train_gen = sup_batch.BatchGeneratorSupervised(batch_size=config.cfg.TRAIN_PARAM.BATCH_SIZE,
                                                   training_size=config.cfg.TRAIN_PARAM.TRAINING_SIZE,
                                                   data_list=train_data_list,
                                                   classes=classes,
                                                   shuffle=True,
                                                   image_normalization_fn=utils.normalize_image_mv2,
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
                                                 image_normalization_fn=utils.normalize_image_mv2,
                                                 label_normalization_fn=utils.normalize_label,
                                                 img_aug_conf=img_aug_conf_valid,
                                                 encoding_fn=utils.encode_5FP
                                                 )

    n_batches_train, _ = divmod(np.shape(train_data_list)[
                                    0], config.cfg.TRAIN_PARAM.BATCH_SIZE)
    n_batches_eval, _ = divmod(np.shape(val_data_list)[
                                   0], config.cfg.TRAIN_PARAM.BATCH_SIZE)

    return train_gen, val_gen, n_batches_train, n_batches_eval
