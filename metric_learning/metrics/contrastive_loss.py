from util.registry.metric import Metric

from collections import defaultdict
from tqdm import tqdm
from util.tensor_operations import compute_pairwise_distances
from util.tensor_operations import pairwise_matching_matrix, pairwise_product
from util.tensor_operations import stable_sqrt
from util.tensor_operations import upper_triangular_part
from metric_learning.constants.distance_function import DistanceFunction

import random
import tensorflow as tf


def count_singletons(all_labels):
    counts = defaultdict(int)
    for label in all_labels:
        counts[label] += 1
    return len(list(filter(lambda x: x == 1, counts.values())))


def compute_one_loss(conf, num_labels, label_counts, data, i, j):
    embeddings, labels = data[i]
    test_embeddings, test_labels = data[j]
    distances = compute_pairwise_distances(
        embeddings, test_embeddings, DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED)
    matches = pairwise_matching_matrix(labels, test_labels)
    label_product = pairwise_product(
        tf.gather(label_counts, labels),
        tf.gather(label_counts, test_labels),
    )
    l = 1 / (conf['loss'].get('l', 256) + 1)
    if i == j:
        label_product = label_product - tf.diag(tf.gather(label_counts, labels))
        positive_distances = tf.boolean_mask(distances, matches)
        negative_distances = tf.boolean_mask(distances, ~matches)
        positive_weights = tf.boolean_mask(l / num_labels / label_product, matches)
        negative_weights = tf.boolean_mask((1 - l) / num_labels / (num_labels - 1) / label_product, ~matches)
    else:
        positive_distances = tf.boolean_mask(distances, matches)
        negative_distances = tf.boolean_mask(distances, ~matches)
        positive_weights = tf.boolean_mask(l / num_labels / label_product, matches)
        negative_weights = tf.boolean_mask((1 - l) / num_labels / (num_labels - 1) / label_product, ~matches)
    loss_value = (
        sum(positive_distances * positive_weights) +
        sum(negative_weights * tf.square(tf.maximum(0, conf['loss']['alpha'] - stable_sqrt(negative_distances))))
    )
    return loss_value

def compute_contrastive_loss(conf, embeddings_list, labels_list, extra_info):
    label_counts = tf.constant(extra_info['label_counts'], dtype=tf.float32)
    num_labels = extra_info['num_labels']
    data = list(zip(embeddings_list, labels_list))
    positive_loss = 0.0
    negative_loss = 0.0
    positive_indices = list(zip(range(len(data)), range(len(data))))
    negative_indices = [random.sample(list(range(len(data))), 2) for i in range(1000)]
    for i, j in tqdm(positive_indices, total=len(positive_indices), desc='positive loss', dynamic_ncols=True):
        positive_loss += compute_one_loss(conf, num_labels, label_counts, data, i, j)
    for i, j in tqdm(negative_indices, total=len(negative_indices), desc='negative loss', dynamic_ncols=True):
        loss = compute_one_loss(conf, num_labels, label_counts, data, i, j)
        if i == len(data) - 1:
            loss *= int(data[0][0].shape[0]) / int(data[-1][0].shape[0])
        if j == len(data) - 1:
            loss *= int(data[0][0].shape[0]) / int(data[-1][0].shape[0])
        negative_loss += compute_one_loss(conf, num_labels, label_counts, data, i, j)
    return float(positive_loss + negative_loss / 1000 * len(data) * (len(data) - 1))


class ContrastiveLoss(Metric):
    name = 'contrastive_loss'

    def compute_metric(self, model, ds, num_testcases):
        embeddings_list, labels_list = self.get_embeddings(
            model, ds, num_testcases)

        return compute_contrastive_loss(
            self.conf,
            embeddings_list,
            labels_list,
            model.extra_info)
