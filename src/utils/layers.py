from tensorflow.keras.layers import Conv2D, SeparableConv2D, Activation, BatchNormalization, concatenate, \
    AveragePooling2D, UpSampling2D, Add


def convolution(x,
                filters,
                kernel_size,
                strides=(1, 1),
                padding='same',
                data_format=None,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                with_batch_norm=True,
                batch_norm_params=None):
    """

    :param x:
    :param filters:
    :param kernel_size:
    :param strides:
    :param padding:
    :param data_format:
    :param activation:
    :param use_bias:
    :param kernel_initializer:
    :param bias_initializer:
    :param kernel_regularizer:
    :param bias_regularizer:
    :param with_batch_norm:
    :param batch_norm_params:
    :return:
    """

    if not with_batch_norm:
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                   data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer)(x)
    else:
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None,
                   data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                   bias_regularizer=bias_regularizer)(x)
        batch_norm_params = {} if batch_norm_params is None else batch_norm_params
        x = BatchNormalization(**batch_norm_params)(x)
        x = Activation(activation=activation)(x)

    return x


def separable_convolution(x,
                          filters,
                          kernel_size,
                          strides=(1, 1),
                          padding='same',
                          data_format=None,
                          activation=None,
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=None,
                          bias_regularizer=None,
                          with_batch_norm=True,
                          batch_norm_params=None):
    """

    :param x:
    :param filters:
    :param kernel_size:
    :param strides:
    :param padding:
    :param data_format:
    :param activation:
    :param use_bias:
    :param kernel_initializer:
    :param bias_initializer:
    :param kernel_regularizer:
    :param bias_regularizer:
    :param with_batch_norm:
    :param batch_norm_params:
    :return:
    """

    if not with_batch_norm:
        x = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            activation=activation, data_format=data_format, use_bias=use_bias,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(x)
    else:
        x = SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None,
                            data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer)(x)
        batch_norm_params = {} if batch_norm_params is None else batch_norm_params
        x = BatchNormalization(**batch_norm_params)(x)
        x = Activation(activation=activation)(x)

    return x


def dense_block(x,
                kernel_size,
                num_layers=2,
                growth_rate=32,
                filters_inter=None,
                data_format=None,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                with_batch_norm=True,
                batch_norm_params=None):
    """

    :param x:
    :param filters:
    :param kernel_size:
    :param strides:
    :param num_layers:
    :param growth_rate:
    :param filters_inter:
    :param padding:
    :param data_format:
    :param activation:
    :param use_bias:
    :param kernel_initializer:
    :param bias_initializer:
    :param kernel_regularizer:
    :param bias_regularizer:
    :param with_batch_norm:
    :param batch_norm_params:
    :return:
    """

    use_bias = use_bias and (False if with_batch_norm else True)

    for l in range(num_layers):
        if activation is not None:
            x0 = Activation(activation)(x)
        else:
            x0 = x

        if filters_inter is not None:
            if not with_batch_norm:
                x0 = Conv2D(filters=filters_inter, kernel_size=(1, 1), padding='same', activation=activation,
                            data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer)(x0)
            else:
                x0 = Conv2D(filters=filters_inter, kernel_size=(1, 1), padding='same', activation=None,
                            data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer)(x0)
                batch_norm_params = {} if batch_norm_params is None else batch_norm_params
                x0 = BatchNormalization(**batch_norm_params)(x0)
                x0 = Activation(activation=activation)(x0)

        if not with_batch_norm:
            x0 = Conv2D(filters=growth_rate, kernel_size=kernel_size, padding='same', activation=activation,
                        data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)(x0)
        else:
            x0 = Conv2D(filters=growth_rate, kernel_size=kernel_size, padding='same', activation=None,
                        data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)(x0)
            batch_norm_params = {} if batch_norm_params is None else batch_norm_params
            x0 = BatchNormalization(**batch_norm_params)(x0)
            x0 = Activation(activation=activation)(x0)

        if data_format is None or data_format.lower() == 'channels_last':
            x = concatenate([x, x0])
        else:
            x = concatenate([x, x0], axis=1)

    return x


def separable_dense_block(x,
                          kernel_size,
                          num_layers=2,
                          growth_rate=32,
                          filters_inter=None,
                          data_format=None,
                          activation=None,
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=None,
                          bias_regularizer=None,
                          with_batch_norm=True,
                          batch_norm_params=None):
    """

    :param x:
    :param filters:
    :param kernel_size:
    :param strides:
    :param num_layers:
    :param growth_rate:
    :param filters_inter:
    :param padding:
    :param data_format:
    :param activation:
    :param use_bias:
    :param kernel_initializer:
    :param bias_initializer:
    :param kernel_regularizer:
    :param bias_regularizer:
    :param with_batch_norm:
    :param batch_norm_params:
    :return:
    """

    use_bias = use_bias and (False if with_batch_norm else True)

    for l in range(num_layers):
        if activation is not None:
            x0 = Activation(activation)(x)
        else:
            x0 = x

        if filters_inter is not None:
            if not with_batch_norm:
                x0 = Conv2D(filters=filters_inter, kernel_size=(1, 1), padding='same', activation=activation,
                            data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer)(x0)
            else:
                x0 = Conv2D(filters=filters_inter, kernel_size=(1, 1), padding='same', activation=None,
                            data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer)(x0)
                batch_norm_params = {} if batch_norm_params is None else batch_norm_params
                x0 = BatchNormalization(**batch_norm_params)(x0)
                x0 = Activation(activation=activation)(x0)

        if not with_batch_norm:
            x0 = SeparableConv2D(filters=growth_rate, kernel_size=kernel_size, padding='same', activation=activation,
                                 data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer)(x0)
        else:
            x0 = SeparableConv2D(filters=growth_rate, kernel_size=kernel_size, padding='same', activation=None,
                                 data_format=data_format, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer)(x0)
            batch_norm_params = {} if batch_norm_params is None else batch_norm_params
            x0 = BatchNormalization(**batch_norm_params)(x0)
            x0 = Activation(activation=activation)(x0)

        if data_format is None or data_format.lower() == 'channels_last':
            x = concatenate([x, x0], axis=-1)
        else:
            x = concatenate([x, x0], axis=1)

    return x
