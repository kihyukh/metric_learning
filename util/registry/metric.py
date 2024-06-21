from util.registry.class_registry import ClassRegistry

from tqdm import tqdm

import math

class Metric(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.metrics'

    cache = {}

    def __init__(self, conf, extra_info):
        super(Metric, self).__init__()
        self.conf = conf
        self.metric_conf = None
        for metric in conf['metrics']:
            if metric['name'] == self.name:
                self.metric_conf = metric
                break
        self.extra_info = extra_info

    def get_embeddings(self, model, dataset, num_testcases):
        dataset_name = self.metric_conf['dataset']
        if dataset_name in Metric.cache:
            return Metric.cache[dataset_name]
        batch_size = self.metric_conf['batch_design']['batch_size']
        dataset = dataset.batch(batch_size)
        batches = tqdm(
            dataset,
            total=math.ceil(num_testcases / batch_size),
            desc='embedding',
            dynamic_ncols=True)
        embeddings_list = []
        labels_list = []
        for images, labels in batches:
            embeddings = model(images, training=False)
            embeddings_list.append(embeddings)
            labels_list.append(labels)
        Metric.cache[dataset_name] = (embeddings_list, labels_list)
        return Metric.cache[dataset_name]

    def compute_metric(self, model, test_ds, num_testcases):
        raise NotImplementedError

    def __str__(self):
        return self.conf['name']
