import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dropout, Dense
import src.utils.config as config


def mobilenet_v2_custom(num_output=config.cfg.TRAIN_PARAM.NUM_PARAMETERS, training_size=config.cfg.TRAIN_PARAM.
                        TRAINING_SIZE):
    base_model = MobileNetV2(input_shape=(training_size, training_size, 3),
                             include_top=False,
                             weights='imagenet')

    for layer in base_model.layers[:3]:
        layer.trainable = False

    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Dense(num_output, activation='linear'))
    model.summary()
    return model


def vgg19_custom(num_output=config.cfg.TRAIN_PARAM.NUM_PARAMETERS, training_size=config.cfg.TRAIN_PARAM.TRAINING_SIZE):
    base_model = VGG19(input_shape=(training_size, training_size, 3),
                       include_top=False,
                       weights='imagenet')

    for layer in base_model.layers[:20]:
        layer.trainable = False

    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(num_output, activation='linear'))
    model.summary()
    return model
