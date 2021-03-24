import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dropout, Dense
import src.utils.config as config


def mobilenet_v2_custom(num_output=config.cfg.TRAIN_PARAM.NUM_PARAMETERS, training_size=config.cfg.TRAIN_PARAM.
                        TRAINING_SIZE):
    base_model = MobileNetV2(input_shape=(training_size, training_size, 3),
                             include_top=False,
                             weights='imagenet')

    # Ne pas entraîner les 5 premières couches (les plus basses)
    base_model.summary()
    for layer in base_model.layers[:5]:
        layer.trainable = False

    model = tf.keras.Sequential()
    model.add(base_model)
    # model.add(GlobalAveragePooling2D())
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_output, activation='linear'))
    model.summary()
    return model


def vgg19_custom(num_output=config.cfg.TRAIN_PARAM.NUM_PARAMETERS, training_size=config.cfg.TRAIN_PARAM.TRAINING_SIZE):
    base_model = VGG19(input_shape=(training_size, training_size, 3),
                       include_top=False,
                       weights='imagenet')

    nb_layer = 0
    for layer in base_model.layers:
        nb_layer += 1
        if nb_layer < 20: layer.trainable = False
    print('nb_layer', nb_layer)

    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(num_output, activation='linear'))
    model.summary()
    return model

if __name__ == '__main__':
    vgg19_custom()
