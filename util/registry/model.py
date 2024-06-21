import tensorflow as tf

from util.registry.class_registry import ClassRegistry
from util.registry.loss_function import LossFunction
from tensorflow.keras.layers import Dense


class Model(tf.keras.models.Model, metaclass=ClassRegistry):
    module_path = 'metric_learning.models'

    loss_function = None
    model = None

    variable_names = ['model']

    def __init__(self, conf, extra_info):
        super(Model, self).__init__()
        self.conf = conf
        self.extra_info = extra_info

        self.loss_function = LossFunction.create(conf['loss']['name'], conf, extra_info)
        for k, v in self.loss_function.extra_variables.items():
            setattr(self, k, v)
            self.variable_names.append(k)
        if 'dimension' in conf['model']:
            self.embedding = Dense(conf['model']['dimension'],
                                   name='dimension_reduction')
            self.variable_names.append('embedding')

    def loss(self, batch, model, dataset):
        return self.loss_function.loss(batch, model, dataset)

    def __str__(self):
        return self.conf['model']['name'] + '_' + str(self.loss_function)

    def preprocess_image(self, image):
        return (image / 255. - 0.5) * 2

    def learning_rates(self):
        ret = {}
        for variable_name in self.variable_names:
            lr = self.conf['trainer'].get('lr_{}'.format(variable_name),
                                          self.conf['trainer']['learning_rate'])
            variable = getattr(self, variable_name)
            if self.conf['trainer'].get('lr_decay_steps'):
                lr = tf.train.exponential_decay(
                    lr,
                    global_step=tf.train.get_or_create_global_step(),
                    decay_steps=self.conf['trainer']['lr_decay_steps'],
                    decay_rate=self.conf['trainer']['lr_decay_rate'],
                    staircase=True
                )
            if hasattr(variable, 'variables'):
                ret[variable_name] = (lr, variable.variables)
            else:
                ret[variable_name] = (lr, [variable])
        return ret

    def call(self, inputs, training=None, mask=None):
        ret = self.model(self.preprocess_image(inputs),
                         training=training,
                         mask=mask)
        if 'dimension' in self.conf['model']:
            ret = self.embedding(ret)
        if self.conf['model']['l2_normalize']:
            ret = tf.nn.l2_normalize(ret)
        return ret
