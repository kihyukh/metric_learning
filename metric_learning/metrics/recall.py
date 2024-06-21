from util.registry.metric import Metric

from collections import defaultdict
from tqdm import tqdm
from util.tensor_operations import compute_pairwise_distances
from metric_learning.constants.distance_function import get_distance_function

import tensorflow as tf


def count_singletons(all_labels):
    counts = defaultdict(int)
    for label in all_labels:
        counts[label] += 1
    return len(list(filter(lambda x: x == 1, counts.values())))


def compute_recall(embeddings_list, labels_list, k_list, distance_function):
    successes = defaultdict(float)
    total = 0.
    num_singletons = 0
    data = list(zip(embeddings_list, labels_list))
    batches = tqdm(
        data, total=len(embeddings_list), desc='recall', dynamic_ncols=True)
    for i, (embeddings, labels) in enumerate(batches):
        all_labels = []
        distance_blocks = []
        for j, (test_embeddings, test_labels) in enumerate(data):
            all_labels += list(test_labels.numpy())
            pairwise_distances = compute_pairwise_distances(
                embeddings, test_embeddings, distance_function)
            if i == j:
                pairwise_distances = pairwise_distances + \
                         tf.eye(int(pairwise_distances.shape[0])) * 1e6
            distance_blocks.append(pairwise_distances)

        values, indices = tf.nn.top_k(-tf.concat(distance_blocks, axis=1), max(k_list))
        top_labels = tf.gather(tf.constant(all_labels, tf.int64), indices)
        for k in k_list:
            score = tf.reduce_sum(
                tf.cast(tf.equal(
                    tf.transpose(labels[None]),
                    top_labels[:, 0:k]
                ), tf.int32), axis=1)
            successes[k] += int(sum(tf.cast(score >= 1, tf.int32)))
        total += int(embeddings.shape[0])
        num_singletons = count_singletons(all_labels)
    return {k: success / float(total - num_singletons) for k, success in successes.items()}


class Recall(Metric):
    name = 'recall'

    def compute_metric(self, model, ds, num_testcases):
        embeddings_list, labels_list = self.get_embeddings(model, ds, num_testcases)

        ret = compute_recall(
            embeddings_list,
            labels_list,
            self.metric_conf['k'],
            get_distance_function(self.conf['loss']['distance_function']))
        return {'recall@{}'.format(k): score for k, score in ret.items()}
