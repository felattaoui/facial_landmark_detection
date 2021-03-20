#!python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import numpy as np
import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import regularizers, initializers

# class size( np.int64 ):
#     """ define a size class to allow custom formatting
#         Implements a format specifier of S for the size class - which displays a human readable in b, kb, Mb etc
#         from http://code.activestate.com/recipes/578321-human-readable-filememory-sizes/
#     """
#     def __format__(self, fmt):
#         if fmt == "" or fmt[-1] != "S":
#             if fmt[-1].tolower() in ['b','c','d','o','x','n','e','f','g','%']:
#                 # Numeric format.
#                 return np.int64(self).__format__(fmt)
#             else:
#                 return str(self).__format__(fmt)
#
#         val, s = float(self), ["b ","Kb","Mb","Gb","Tb","Pb"]
#         if val<1:
#             # Can't take log(0) in any base.
#             i,v = 0,0
#         else:
#             i = int(math.log(val,1024))+1
#             v = val / math.pow(1024,i)
#             v,i = (v,i) if v > 0.5 else (v*1024,i-1)
#         return ("{0:{1}f} "+s[i]).format(v, fmt[:-1])
#
#
# def __get_flops():
#     """
#     This function does not really work properly
#     From: https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras
#     :return: The total number of thousand FLOPs of the currently compile model.
#     """
#     run_meta = tf.RunMetadata()
#     opts = (tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.float_operation())
#             .with_empty_output()
#             .build())
#
#     # We use the Keras session graph in the call to the profiler.
#     flops = tf.profiler.profile(graph=K.get_session().graph, options=opts,
#             run_meta=run_meta, cmd='op')
#
#     return flops.total_float_ops  # Prints the "flops" of the model.
#
# def get_flops(model, verbose=True, print_fn=None):
#     """
#     Count the number of FLOPs and of weights of a model
#     :param model: Keras model
#     :param verbose: If True, the number of FLOPs for each layer is printed
#     :return: a tuple with the nuber of Mega FLOPs and the number of weights
#     """
#     if print_fn is None:
#         print_fn=print
#     if verbose:
#         print_fn('%3s %-30s %25s %12s %9s' % ('N.', 'Name', 'Info', 'Weights', 'FLOPs (M)'))
#     tot_weight = tot_flop = np.int64(0)
#     # model.layers[0].input_shape = model.layers[0].input_shape[0]
#     # model.layers[0].output_shape = model.layers[0].output_shape[0]
#     for layer_id, layer in enumerate(model.layers):
#         info = ''
#         # if isinstance(layer, (keras.layers.Flatten, keras.layers.Dropout)):
#         #     continue
#         if not hasattr(layer, 'input_shape') or not hasattr(layer, 'output_shape'):
#             continue
#         h_i = w_i = c_i = h_o = w_o = c_o = 1
#         if len(layer.input_shape) == 4:
#             h_i = layer.input_shape[1]
#             w_i = layer.input_shape[2]
#             c_i = layer.input_shape[3]
#         else:
#             h_i = layer.input_shape[1]
#         if len(layer.output_shape) == 4:
#             h_o = layer.output_shape[1]
#             w_o = layer.output_shape[2]
#             c_o = layer.output_shape[3]
#         else:
#             h_o = layer.output_shape[1]
#         n_weight = flop = 0
#         if isinstance(layer, keras.layers.DepthwiseConv2D):
#             w_shape = layer.get_weights()[0].shape
#             n_weight = c_i * np.product(w_shape[0:2])
#             flop = 2 * h_o * w_o * c_i * np.product(w_shape[0:2])
#             info = '%dx%d, %d' % (w_shape[0], w_shape[1], w_shape[2])
#             if layer.use_bias:
#                 n_weight += c_o
#                 flop += h_o + w_o + c_o
#         elif isinstance(layer, keras.layers.Conv2D):
#             # n. flop Conv2D: c_o * (k_s * k_s * c_i) * (w_o * h_o) * 2
#             w_shape = layer.get_weights()[0].shape
#             n_weight = c_o * c_i * np.product(w_shape[0:2])
#             flop = 2 * h_o * w_o * n_weight
#             info = '%dx%d, %d' % (w_shape[0], w_shape[1], w_shape[2])
#             if layer.use_bias:
#                 n_weight += c_o
#                 flop += h_o + w_o + c_o
#         elif isinstance(layer, keras.layers.SeparableConv2D):
#             w_shape = layer.get_weights()[0].shape
#             n_weight = c_i * np.product(w_shape[0:2]) + c_i * c_o
#             flop = 2 * h_o * w_o * (c_i * np.product(w_shape[0:2]) + c_i * c_o)
#             info = '%dx%d, %d' % (w_shape[0], w_shape[1], w_shape[2])
#             if layer.use_bias:
#                 n_weight += c_o
#                 flop += h_o + w_o + c_o
#         elif isinstance(layer, keras.layers.Dense):
#             # n. flop Conv2D: c_o * (k_s * k_s * c_i) * (w_o * h_o) * 2
#             n_weight = h_i * h_o
#             flop = 2 * h_i * h_o
#             if layer.use_bias:
#                 n_weight += h_o
#                 flop += h_o
#         elif isinstance(layer, keras.layers.BatchNormalization):
#             n_weight = 2 * c_o
#             flop = 2 * h_o * w_o * c_o
#         elif isinstance(layer, keras.layers.Activation):
#             n_weight = 0
#             flop = h_o * w_o * c_o
#         elif isinstance(layer, keras.layers.Add):
#             n_weight = 0
#             flop = h_o * w_o * c_o
#         tot_weight += n_weight
#         tot_flop += flop
#         if verbose:
#             print_fn('{:>3} {:<30} {:>25} {:>12,} {:9.3f}'.format(layer_id, layer.name, info, n_weight, flop/1000000))
#     if verbose:
#         print_fn('Total params: {:,}'.format(tot_weight))
#         print_fn('Memory required: {0:.2S}'.format(size(4*tot_weight)))
#         print_fn('Total FLOPs: %.3f Mflop' % (tot_flop/1000000))
#     return tot_flop/1000000, tot_weight
#
#
# def build_network_conv_only(input_shape, conv_filters, num_outputs=1,
#                             dropout_before_last_dense=0.0,
#                             global_average_pooling_before_last_dense=True, param_fn=None):
#     """
#     Build a network with convolutions only. The stride is 2 for each convolution. There is a final dense layer.
#     :param input_shape: (height, width, channels)
#     :param conv_filters: List of filters for each convolution.
#     :param num_outputs: output of thefinal dense layer. If null, there is no dense layer.
#     :dropout_before_last_dense: If non-null add a drop out before then dense layer.
#     :return: Keras model
#     """
#
#     inputs = Input(shape=input_shape)
#     x = inputs
#     for f in conv_filters:
#         x = conv(x, filters=f, kernel_size=(3, 3), strides=2, use_bias=False, param_fn=param_fn)
#         x = bn(x, param_fn)
#         x = act(x, param_fn)
#     if num_outputs > 0:
#         x = last_dense(x, num_outputs, dropout_rate=dropout_before_last_dense,
#                        global_average_pooling=global_average_pooling_before_last_dense,
#                        param_fn=param_fn)
#     return Model(inputs, x)
#
