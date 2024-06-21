import tensorflow as tf

from util.registry.model import Model


layers = tf.keras.layers


class SimpleConvolutionModel(Model):
    name = 'simple_conv'

    def __init__(self, conf, extra_info):
        super(SimpleConvolutionModel, self).__init__(conf, extra_info)

        data_format = 'channels_last'

        max_pool = layers.MaxPooling2D(
            (2, 2), (2, 2), padding='same', data_format=data_format)

        self.model = tf.keras.Sequential(
            [
                layers.Conv2D(
                    32,
                    3,
                    padding='same',
                    data_format=data_format,
                    activation=tf.nn.relu),
                max_pool,
                layers.Conv2D(
                    64,
                    3,
                    padding='same',
                    data_format=data_format,
                    activation=tf.nn.relu),
                max_pool,
                layers.Flatten(),
                layers.Dense(128, activation=tf.nn.relu),
                layers.Dropout(0.4),
                layers.Dense(conf['k']),
            ])
