from util.registry.class_registry import ClassRegistry


class LossFunction(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.loss_functions'

    def __init__(self, conf, extra_info):
        super(LossFunction, self).__init__()
        self.conf = conf
        self.extra_info = extra_info
        self.extra_variables = {}

    def loss(self, batch, model, dataset):
        raise NotImplementedError

    def __str__(self):
        return self.name
