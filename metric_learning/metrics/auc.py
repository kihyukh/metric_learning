from util.registry.metric import Metric

from collections import defaultdict
from metric_learning.constants.distance_function import get_distance_function
from util.tensor_operations import compute_elementwise_distances

import random
import tensorflow as tf


def compute_distances(embeddings, first_list, second_list, distance_function):
    first_embeddings = tf.gather(embeddings, first_list)
    second_embeddings = tf.gather(embeddings, second_list)
    return compute_elementwise_distances(
        first_embeddings, second_embeddings, distance_function)


class AUC(Metric):
    name = 'auc'

    def compute_metric(self, model, ds, num_testcases):
        embeddings_list, labels_list = self.get_embeddings(
            model, ds, num_testcases)
        distance_function = get_distance_function(
            self.conf['loss']['distance_function'])

        embeddings = tf.concat(embeddings_list, axis=0)
        labels = tf.concat(labels_list, axis=0)

        label_map = defaultdict(list)
        for index, label in enumerate(labels):
            label_map[int(label)].append(index)
        num_samples = self.metric_conf['num_samples']
        positive_label_map = dict(
            [(k, v) for k, v in label_map.items() if len(v) >= 2]
        )

        positive_labels = random.choices(
            population=list(positive_label_map.keys()),
            weights=[len(v) for v in positive_label_map.values()],
            k=num_samples,
        )
        positive_pairs = []
        for positive_label in positive_labels:
            a, b = random.sample(label_map[positive_label], 2)
            positive_pairs.append((a, b))
        first_list, second_list = zip(*positive_pairs)
        positive_distances = compute_distances(
            embeddings, first_list, second_list, distance_function)

        negative_label_pairs = []
        population = list(label_map.keys())
        weights = [len(v) for v in label_map.values()]
        for _ in range(num_samples):
            while True:
                a, b = random.choices(
                    population=population,
                    weights=weights,
                    k=2,
                )
                if a != b:
                    break
            negative_label_pairs.append((a, b))
        negative_pairs = []
        for a, b in negative_label_pairs:
            first = random.choice(label_map[a])
            second = random.choice(label_map[b])
            negative_pairs.append((first, second))
        first_list, second_list = zip(*negative_pairs)
        negative_distances = compute_distances(
            embeddings, first_list, second_list, distance_function)

        return float(tf.reduce_sum(tf.cast(positive_distances < negative_distances, tf.float32))) / num_samples
