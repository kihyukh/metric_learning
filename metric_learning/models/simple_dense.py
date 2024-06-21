import tensorflow as tf

from util.registry.model import Model


class SimpleDenseModel(Model):
    name = 'simple_dense'

    def __init__(self, conf, extra_info):
        super(SimpleDenseModel, self).__init__(conf, extra_info)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(conf['model']['k'])
        ])
