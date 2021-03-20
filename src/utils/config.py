from easydict import EasyDict
import src.utils.utils as utils

__C = EasyDict()

cfg = __C

# create CNN dict
__C.TRAIN_PARAM = EasyDict()

__C.TRAIN_PARAM.TRAINING_SIZE = 96
__C.TRAIN_PARAM.DATASET_SIZE = 10000
__C.TRAIN_PARAM.NUM_PARAMETERS = 10
__C.TRAIN_PARAM.BATCH_SIZE = 32
__C.TRAIN_PARAM.USE_MULTIPROCESSING = False
__C.TRAIN_PARAM.WORKERS = 8

# create Train Path dict
__C.TRAIN_PATH = EasyDict()
__C.TRAIN_PATH.MODEL_NAME = utils.model_name(__C.TRAIN_PARAM.DATASET_SIZE, r'/home/farid/PycharmProjects/facial_landmark_detection/models/supervised/new')
__C.TRAIN_PATH.TENSORBOARD_LOGS = __C.TRAIN_PATH.MODEL_NAME + 'logs'
__C.TRAIN_PATH.KERAS_MODEL = __C.TRAIN_PATH.MODEL_NAME + '.keras.model'
__C.TRAIN_PATH.PATH_MODEL = None
__C.TRAIN_PATH.IMAGE_DIR = r'../../data/'
__C.TRAIN_PATH.TRAIN_TXT_FILE, __C.TRAIN_PATH.VAL_TXT_FILE = utils.get_train_and_valid_file(__C.TRAIN_PARAM.DATASET_SIZE)
