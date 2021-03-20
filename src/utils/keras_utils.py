#!python
# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Activation, Dense, Conv2D, SeparableConv2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, GlobalAveragePooling2D, Lambda, Concatenate
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

from copy import deepcopy


def remove_keys(d, keys):
    """Utility that easily removes one or several keys from a dictionary

    :param d: dictionary. left unchanged by the function.
    :param keys: string of list of string withthe keys to remove. Nothing happens if a string is not in the dictionary.
    :return: the dictionary d without the keys.
    """
    pp = deepcopy(d)
    if isinstance(keys, (list, tuple)):
        for k in keys:
            pp.pop(k, None)
    else:
        pp.pop(keys, None)
    return pp


def get_channel_axis():
    """
    :return: The index of the channel axis in a tensor.
    """
    return 1 if K.image_data_format() == 'channels_first' else 3


def get_spatial_axis():
    """
    :return: The index of the Y and X dimensions in a tensor.
    """
    if K.image_data_format() == 'channels_first':
        return 2, 3
    else:
        return 1, 2


def example_param_fn(layer_type, in_shape=None, **other_params):
    """
    Example of a function that must be implemented by the user of these layer. The purpose of this function is to define
    the default parameter for each layer.

    Each Keras layers accepts a series of parameters. The implementor of a network only sets the parameter that the
    network required, for instance the kernel_size of a Conv2D if the network must perform a pointwize convolution.
    It is up to this function to set the other parameters of the layer. For instance, the kernel_initializer, etc.
    In conclusion, this function allows the user to change the default parameters of each Keras layer.

    :param layer_type: Type of layer, can be 'BatchNormalization', 'Dense', 'Conv2D', 'AveragePooling2D', 'Shortcut',
        'Cardinality'
    :param in_shape: Shape of the input tensor. May be used to adapt the default parameters based on the size of the
        input tensor (based on the number of channels for instance).
    :param other_params: Parameters set by the implementaton of the network. These parameters override the ones returned
        by this function.
    :return: A dictionnary with the parameters of the layer.
    """
    layer_type = layer_type.lower()
    d={}
    if layer_type == 'batchnormalization':
        d['momentum'] = 0.99
        d['epsilon'] = 1e-3
    elif layer_type == 'dense':
        # Le 'gain=2.0' ici (et pour la conv2D ci-dessous) suppose l'utilisation d'un Relu comme activation,
        # Voir l'article 'Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classiﬁcation'
        # Equation 12.
        d['kernel_initializer'] = initializers.Orthogonal(gain=2.0, seed=None)
        d['kernel_regularizer'] = None
    elif layer_type == 'conv2d':
        d['type'] = 'Conv2D'
        d['kernel_size'] = (3,3)    # Kernel size of the spatial convolution (used only if no 'kernel_size' is specified)
        d['padding'] = 'same'
        d['kernel_initializer'] = initializers.Orthogonal(gain=2.0, seed=None)
        d['kernel_regularizer'] = None
    elif layer_type == 'averagepooling2d':
        d['padding'] = 'same'
    elif layer_type == 'shortcut':
        d['use_projection'] = True # Option (B) of paper 'Deep Residual Learning for Image Recognition' Pg 4
    elif layer_type == 'cardinality':
        # Number of convolutions in the grouped convolutions. For each 3x3 convolution,
        #   the number of filters is filters // cardinality
        # Only used for the ResNeXt
        return 4
    return d


def bn(inputs, param_fn=None, **override_params):
    """Batch-Normalization layer
    :param param_fn: Parameter function. See example_param_fn()
    :param override_params: Parameters given to the function keras.layers.BatchNormalization() and
        which supersede the ones returned by param_fn
    """
    pp = deepcopy(override_params)
    if param_fn is not None:
        pp = param_fn('BatchNormalization', K.int_shape(inputs), **override_params)
        pp.update(override_params)
    if 'axis' not in override_params.keys():
        pp['axis'] = get_channel_axis()

    return BatchNormalization(**pp)(inputs)


def act(inputs, param_fn=None, **override_params):
    """Activation layer
    :param param_fn: Parameter function. See example_param_fn()
    :param override_params: Parameters given to the function keras.layers.Activation() and
        which supersede the ones returned by param_fn
    """
    pp = deepcopy(override_params)
    if param_fn is not None:
        pp = param_fn('Activation', K.int_shape(inputs), **override_params)
        pp.update(override_params)
    pp['activation'] = pp.setdefault('activation', 'relu')
    return Activation(**pp)(inputs)

def dense(inputs, units, param_fn=None, **override_params):
    """Dense layer with the following default parameters:
    :param units: Positive integer, dimensionality of the output space.
    :param param_fn: Parameter function. See example_param_fn()
    :param override_params: Parameters given to the function keras.layers.Dense() and
        which supersede the ones returned by param_fn
    """
    pp = deepcopy(override_params)
    if param_fn is not None:
        pp = param_fn('Dense', K.int_shape(inputs), **override_params)
        pp.update(override_params)

    return Dense(units=units, **pp)(inputs)


def conv(inputs, filters, param_fn=None, **override_params):
    """2D convolution layer with the following default parameters:
    :param conv_type: Type of convolution: Default: 'Conv2D', Possible values: 'DepthwiseConv2D',
        'SeparableConv2D'
    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the
        convolution). (Mandatory for Conv2D and SeparableConv2D, not used for DepthwiseConv2D).
    :param param_fn: Parameter function. See example_param_fn()
    :param override_params: Parameters given to the function keras.layers.Conv2D() and
        which supersede the ones returned by param_fn
    """
    pp = deepcopy(override_params)
    if param_fn is not None:
        pp = param_fn('Conv2D', K.int_shape(inputs), **override_params)
        pp.update(override_params)
    if 'type' not in pp.keys():
        pp['type'] = 'Conv2D'
    if 'kernel_size' not in pp.keys():
        pp['kernel_size'] = (3,3)
    conv_type = pp['type'].lower()
    assert conv_type in ('conv2d', 'separableconv2d', 'depthwiseconv2d')
    pp = remove_keys(pp, 'type')

    if conv_type == 'separableconv2d':
        x = SeparableConv2D(filters, **pp)(inputs)
    elif conv_type == 'depthwiseconv2d':
        x = DepthwiseConv2D(**pp)(inputs)
    else:
        x = Conv2D(filters, **pp)(inputs)
    return x

def grouped_conv(inputs, filters, cardinality, param_fn=None, **override_params):
    """2D Grouped Convolution layer.
    Implemented as the block (b) from Figure 3 of paper 'Aggregated Residual Transformations for
    Deep Neural Networks'
    Inspired from: https://github.com/titu1994/Keras-ResNeXt/blob/master/resnext.py

    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters
        in the convolution). The number of filters for each single convolution
        is = filters // cardinality (Mandatory)
    :param cardinality: Number of convolutions. (Mandatory)
    :param param_fn: Parameter function. See example_param_fn()
    :param override_params: Parameters given to the function keras.layers.Conv2D() and
        which supersede the ones returned by param_fn
    """
    channel_axis = get_channel_axis()
    grouped_channels = filters // cardinality
    name = override_params.setdefault('name', None)
    name2 = None

    x = inputs
    if cardinality == 1:
        x = conv(x, filters, param_fn, **override_params)
    else:
        group_list = []
        for c in range(cardinality):
            if name is not None:
                name2 = name.replace('gconv3x3', 'chan_sampling')
            if K.image_data_format() == 'channels_last':
                y = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels],
                           name='%s_%d'%(name2,c))(x)
            else:
                y = Lambda(lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :],
                           name='%s_%d'%(name2,c))(x)

            name2 = None
            if name is not None:
                name2 = name + '_%d' % c
            y = conv(y, grouped_channels, param_fn, name=name2, **remove_keys(override_params, 'name'))

            group_list.append(y)

        if name is not None:
            name2 = name.replace('gconv3x3', 'concat')
        x = Concatenate(axis=channel_axis, name=name2)(group_list)
    return x


def last_dense(inputs, num_outputs, dropout_rate=0.0, global_average_pooling=True, param_fn=None, **override_params):
    """Dense layer with the following default parameters:

    Pseudo-code:

        Input -> GlobalAveragePooling2D -> Flatten -> Drop Out -> Dense

    :param inputs: Input tensor.
    :param num_outputs: Positive integer, dimensionality of the output space.
    :param dropout_rate: float between 0 and 1. Fraction of the input units to drop before the dense layer.
        If 0, the drop out is not done.
    :param global_average_pooling: If True, the feature maps are collapsed to 1 pixel by summation.
    :param param_fn: Parameter function. See example_param_fn()
    :param override_params: Parameters given to the function keras.layers.Dense() and
        which supersede the ones returned by param_fn
    """
    pp = deepcopy(override_params)
    if 'activation' not in pp.keys():
        pp['activation'] = "softmax" if num_outputs > 1 else 'sigmoid'
    assert num_outputs > 0

    x = inputs
    if global_average_pooling:
        x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = dense(x, num_outputs, param_fn=param_fn, **pp)
    return x