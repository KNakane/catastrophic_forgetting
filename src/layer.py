import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import conv_utils

class HyperDense(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        pass

    def call(self, input, weights):
        if self.use_bias:
            kernel, bias = tf.split(weights, [tf.reduce_prod(self.kernel_shape), self.bias_shape])
            kernel = tf.reshape(kernel, self.kernel_shape)
            x = tf.matmul(input, kernel) + bias
        else:
            kernel = tf.split(weights, self.kernel_shape)
            x = tf.matmul(input, kernel)

        return self.activation(x)

    def get_param_shape(self, input_shape):
        input_shape = tf.reduce_prod(input_shape) if isinstance(input_shape, tuple) else input_shape
        self.kernel_shape = [input_shape, self.units]
        self.bias_shape = self.units if self.use_bias else 0
        param_num = input_shape * self.units + self.bias_shape
        return self.kernel_shape, self.bias_shape, param_num

class HyperConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 groups=1,
                 rank=2,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            **kwargs)

        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        pass

    def call(self, input, weights):
        conv_weights = tf.reshape(weights,
                                  (self.kernel_size, self.kernel_size, input.shape[2], self.filters))
        x = tf.nn.conv2d(input,
                         conv_weights,
                         strides=[1, self.strides[0], self.strides[1], 1],
                         padding=self.padding)

        if self.use_bias:
            x += weights

        return self.activation(x)