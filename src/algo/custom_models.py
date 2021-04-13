import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dropout, Dense
import src.utils.config as config


def mobilenet(num_output=config.cfg.TRAIN_PARAM.NUM_PARAMETERS, training_size=config.cfg.TRAIN_PARAM.
              TRAINING_SIZE):
    base_model = tf.keras.applications.MobileNet(input_shape=(training_size, training_size, 3),
                                                 include_top=False,
                                                 weights='imagenet')

    base_model.summary()
    for layer in base_model.layers:
        layer.trainable = True

    model = tf.keras.Sequential()
    model.add(base_model)
    # model.add(GlobalAveragePooling2D())
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_output, activation='linear'))
    model.summary()
    return model


def mobilenetv2(num_output=config.cfg.TRAIN_PARAM.NUM_PARAMETERS, training_size=config.cfg.TRAIN_PARAM.
                TRAINING_SIZE):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(training_size, training_size, 3),
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.summary()
    for layer in base_model.layers:
        layer.trainable = True

    model = tf.keras.Sequential()
    model.add(base_model)
    # model.add(GlobalAveragePooling2D())
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_output, activation='linear'))
    model.summary()
    return model


def vgg19(num_output=config.cfg.TRAIN_PARAM.NUM_PARAMETERS, training_size=config.cfg.TRAIN_PARAM.TRAINING_SIZE):
    base_model = tf.keras.applications.VGG19(input_shape=(training_size, training_size, 3),
                                             include_top=False,
                                             weights='imagenet')

    # nb_layer = 0
    # for layer in base_model.layers:
    #     nb_layer += 1
    #     if nb_layer < 20: layer.trainable = False
    # print('nb_layer', nb_layer)

    base_model.summary()
    for layer in base_model.layers:
        layer.trainable = True

    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(num_output, activation='linear'))
    model.summary()
    return model

# if __name__ == '__main__':
#     vgg19_custom()
#     mobilenet_v2_custom()
