from datetime import datetime
import os
import sys
from tensorflow.python.keras.regularizers import l2
from src.utils.keras_utils import conv, last_dense
from src.utils.layers import separable_dense_block, separable_convolution
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import math
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dropout, Dense

def model_name(size_of_train, logdir='./models/'):
    dt = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
    dt2 = '_train_%d_valid_5000' % size_of_train
    dt = dt + dt2
    return os.path.join(logdir, os.path.splitext(os.path.basename(sys.argv[0]))[0] + dt)


def get_train_and_valid_file(size_of_train):
    train_txt_file = r'../../data/train_%d_5FP.txt' % size_of_train
    val_txt_file = r'../../data/valid_5000_5FP.txt'
    if not os.path.isfile(train_txt_file):
        print("Train file does not exist")

    if not os.path.isfile(val_txt_file):
        print("Validation file does  not exist")

    return train_txt_file, val_txt_file


# def reformat_batch_data(data_list, fusion_op=np.stack):
#     assert (isinstance(data_list, list) and len(data_list) > 0)
#     if isinstance(data_list[0], dict):
#         d = data_list[0]
#         for v in d:
#             d[v] = fusion_op([dd[v] for dd in data_list])
#     elif isinstance(data_list[0], list):
#         d = data_list[0]
#         for i in range(len(d)):
#             d[i] = fusion_op([dd[i] for dd in data_list])
#     else:
#         d = fusion_op(data_list)
#     return d


def get_one_epoch(sequence):
    ims = []
    lbls = []
    for i in range(len(sequence)):
        imb, lblb = sequence.__getitem__(i)
        ims.append(imb)
        lbls.append(lblb)
    return reformat_batch_data(ims, np.concatenate), reformat_batch_data(lbls, np.concatenate)


def mobilenet_custom(num_output):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = MobileNetV2(input_shape=(96, 96, 3),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    outputs = Dense(num_output, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def graph_dense_fp(input_tensor, im_size, num_output):
    x = conv(input_tensor, filters=64, kernel_size=(3, 3), padding='same',
             activation='relu', kernel_regularizer=l2(1e-5))
    x = separable_convolution(x, filters=64, kernel_size=(3, 3), padding='same',
                              activation='relu', strides=(2, 2), kernel_regularizer=l2(1e-5))
    # dense blocks 1
    x = separable_dense_block(x, kernel_size=(3, 3), num_layers=3, growth_rate=12, filters_inter=16, activation='relu',
                              with_batch_norm=True, kernel_regularizer=l2(1e-5))
    x = separable_convolution(x, filters=128, kernel_size=(1, 1), strides=(2, 2), padding='same', activation=None,
                              with_batch_norm=True, kernel_regularizer=l2(1e-5))
    # dense blocks 2
    x = separable_dense_block(x, kernel_size=(3, 3), num_layers=6, growth_rate=12, filters_inter=32, activation='relu',
                              with_batch_norm=True, kernel_regularizer=l2(1e-5))
    x = separable_convolution(x, filters=256, kernel_size=(1, 1), strides=(2, 2), padding='same', activation=None,
                              with_batch_norm=True, kernel_regularizer=l2(1e-5))
    # dense blocks 3
    x = separable_dense_block(x, kernel_size=(3, 3), num_layers=6, growth_rate=12, filters_inter=32, activation='relu',
                              with_batch_norm=True, kernel_regularizer=l2(1e-5))
    x = separable_convolution(x, filters=256, kernel_size=(1, 1), strides=(2, 2), padding='same', activation=None,
                              with_batch_norm=True, kernel_regularizer=l2(1e-5))
    # dense blocks 4
    x = separable_dense_block(x, kernel_size=(3, 3), num_layers=12, growth_rate=24, filters_inter=64, activation='relu',
                              with_batch_norm=True, kernel_regularizer=l2(1e-5))
    x = separable_convolution(x, filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None,
                              with_batch_norm=False, use_bias=True)
    x = conv(x, filters=512, kernel_size=(1, 1), padding='valid', activation='relu',
             use_bias=True)
    x = last_dense(x, num_output, activation=None, global_average_pooling=False)
    return x


# def CNN_test(input_tensor, im_size, num_output):
#     # Start
#     # x = Input(shape=input_shape)
#     nb_filters = 16
#     batch_axis = 3
#
#     layer_1 = Conv2D(nb_filters, (7, 7), strides=1, padding='same', use_bias=False)(input_tensor)
#     layer_1 = BatchNormalization(axis=batch_axis, scale=False)(layer_1)
#     layer_1 = MaxPooling2D(pool_size=(3, 3))(layer_1)
#     layer_1 = Flatten()(layer_1)
#     x = last_dense(layer_1, num_output, activation=None, global_average_pooling=False)
#     return x

def func_keypoints_redefine_boxes(keypoints_on_images, random_state, parents, hooks):
    nb_images = len(keypoints_on_images)
    for i, keypoints_on_image in enumerate(keypoints_on_images):
        l = keypoints_on_image.keypoints
        if (len(l) == 4):
            c = ia.Keypoint(x=sum([l[i].x for i in range(4)]) / 4, y=sum([l[i].y for i in range(4)]) / 4)
            d = (math.hypot(l[0].x - l[2].x, l[0].y - l[2].y) + math.hypot(l[1].x - l[3].x, l[1].y - l[3].y)) / (
                    4 * (math.sqrt(2)))
            l[0] = ia.Keypoint(x=c.x - d, y=c.y - d)
            l[1] = ia.Keypoint(x=c.x - d, y=c.y + d)
            l[2] = ia.Keypoint(x=c.x + d, y=c.y + d)
            l[3] = ia.Keypoint(x=c.x + d, y=c.y - d)
            keypoints_on_image.keypoints = l
    return keypoints_on_images


def func_keypoints_sym_fp(keypoints_on_images, random_state, parents, hooks):
    for i, keypoints_on_image in enumerate(keypoints_on_images):
        l = keypoints_on_image.keypoints
        if (len(l) == 5):
            l[0], l[1] = l[1], l[0]
            l[3], l[4] = l[4], l[3]
            keypoints_on_image.keypoints = l
    return keypoints_on_images


def func_images_void(images, random_state, parents, hooks):
    return images


#
# def func_heatmaps_void(heatmaps, random_state, parents, hooks):
#     return heatmaps


def Sym_5fp():
    return iaa.Lambda(
        func_keypoints=func_keypoints_sym_fp,
        func_images=func_images_void)


def Fliplr_5fp():
    return iaa.Sequential([iaa.Fliplr(1.0), Sym_5fp()])


def RedefineBoxes():
    return iaa.Lambda(
        func_keypoints=func_keypoints_redefine_boxes,
        func_images=func_images_void)


# def _points_to_label(points, image_shape):
#     return [x for xi in points for x in xi]

def normalize_image(img):
    return img / float(255.0)


def normalize_label(labels, training_size):
    c = np.array(10 * [1 / float(training_size)])
    return labels * c


def encode_5FP(img, label):
    return np.array(img), np.array([x for xi in [list(k) for k in label[0]['5FP']] for x in xi])


def load_txt_5FP_and_box(txt_file, image_dir):
    data_list = []
    print('loading annotations')
    raw = list(np.loadtxt(txt_file, dtype=bytes).astype(str))
    for line in raw:
        x1 = float(line[12])
        x2 = float(line[12]) + float(line[14])
        y1 = float(line[13])
        y2 = float(line[13]) + float(line[15])
        data_list += [{
            'filename': image_dir + line[0],
            'objects': [
                {'class': 'face', '5FP': [(float(line[2 * i + 2]), float(line[2 * i + 3])) for i in range(5)],
                 'face_box': [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]}
            ]
        }]
    return data_list
