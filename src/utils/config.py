from easydict import EasyDict
import src.utils.utils as utils
import os

__C = EasyDict()

cfg = __C

# create CNN dict
__C.TRAIN_PARAM = EasyDict()

__C.TRAIN_PARAM.TRAINING_SIZE = 96
__C.TRAIN_PARAM.DATASET_SIZE = 10000
__C.TRAIN_PARAM.NUM_PARAMETERS = 10
__C.TRAIN_PARAM.BATCH_SIZE = 32
__C.TRAIN_PARAM.USE_MULTIPROCESSING = True
__C.TRAIN_PARAM.WORKERS = 4
__C.TRAIN_PARAM.NB_EPOCH = 200

# create Train Path dict
__C.TRAIN_PATH = EasyDict()
__C.TRAIN_PATH.MODEL_NAME_MV2 = utils.model_name(__C.TRAIN_PARAM.DATASET_SIZE, r'../../models/mobilenet_v2/100000_images_no_dataaug_imagenet')
__C.TRAIN_PATH.MODEL_NAME_VGG19 = utils.model_name(__C.TRAIN_PARAM.DATASET_SIZE, r'../../models/vgg19/all_data_aug_imagenet/')
__C.TRAIN_PATH.TENSORBOARD_LOGS_MV2 = __C.TRAIN_PATH.MODEL_NAME_MV2 + '.logs'
__C.TRAIN_PATH.KERAS_MODEL_MV2 = __C.TRAIN_PATH.MODEL_NAME_MV2 + '.keras.model'
__C.TRAIN_PATH.TENSORBOARD_LOGS_VGG19 = __C.TRAIN_PATH.MODEL_NAME_VGG19 + '.logs'
__C.TRAIN_PATH.KERAS_MODEL_VGG19 = __C.TRAIN_PATH.MODEL_NAME_VGG19 + '.keras.model'
__C.TRAIN_PATH.PATH_MODEL = None
__C.TRAIN_PATH.IMAGE_DIR = r'../../data/'
__C.TRAIN_PATH.TRAIN_TXT_FILE, __C.TRAIN_PATH.VAL_TXT_FILE = utils.get_train_and_valid_file(__C.TRAIN_PARAM.DATASET_SIZE)
