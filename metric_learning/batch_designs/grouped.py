from util.registry.batch_design import BatchDesign

from collections import defaultdict
from metric_learning.constants.distance_function import DistanceFunction
from tqdm import tqdm
from util.registry.data_loader import DataLoader
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import get_n_blocks
from util.tensor_operations import pairwise_product
from util.tensor_operations import pairwise_sum
from util.tensor_operations import compute_pairwise_distances
from util.tensor_operations import upper_triangular_part
from util.tensor_operations import stable_sqrt

import math
import numpy as np
import tensorflow as tf
import random


def get_npair_distances(embeddings, n, distance_function, transpose=False):
    num_groups = int(embeddings.shape[0]) // 2
    evens = tf.range(num_groups, dtype=tf.int64) * 2
    odds = tf.range(num_groups, dtype=tf.int64) * 2 + 1
    even_embeddings = tf.gather(embeddings, evens)
    odd_embeddings = tf.gather(embeddings, odds)

    pairwise_distances = compute_pairwise_distances(
        even_embeddings, odd_embeddings, distance_function)

    return (
        get_n_blocks(pairwise_distances, n, transpose=transpose),
        get_n_blocks(
            tf.cast(tf.eye(num_groups), tf.bool), n, transpose=transpose)
    )


class GroupedBatchDesign(BatchDesign):
    name = 'grouped'

    cache = {}

    def create_dataset(self, model, image_files, labels, batch_conf,
                       testing=False):
        data_map = defaultdict(int)
        for image_file, label in zip(image_files, labels):
            data_map[label] += 1
        min_images_per_class = max(
            self.conf['dataset'].get('min_images_per_class', 1),
            self.conf['batch_design']['group_size'])
        data_map = dict(filter(lambda x: x[1] >= min_images_per_class, data_map.items()))
        model.extra_info['num_images'] = sum([y for x, y in data_map.items()])
        model.extra_info['num_labels'] = len(data_map)
        if batch_conf.get('negative_class_mining'):
            data_loader = DataLoader.create(self.conf['dataset']['name'],
                                            self.conf)
            batch_design = BatchDesign.create(
                'vanilla',
                self.conf,
                {'data_loader': data_loader})
            batch_size = 48
            test_dataset, num_testcases = batch_design.create_dataset(
                model, image_files, labels,
                {'name': 'vanilla', 'batch_size': batch_size},
                testing=True)
            test_dataset = test_dataset.batch(batch_size)
            batches = tqdm(
                test_dataset,
                total=math.ceil(num_testcases / batch_size),
                desc='embedding',
                dynamic_ncols=True)
            embeddings_list = []
            labels_list = []
            for mini_images, mini_labels in batches:
                embeddings = model(mini_images, training=False)
                embeddings_list.append(embeddings)
                labels_list.append(mini_labels)
            full_embeddings = tf.concat(embeddings_list, axis=0)
            full_labels = tf.concat(labels_list, axis=0)
            means = []
            mean_distances = []
            num_labels = model.extra_info['num_labels']
            for label in range(num_labels):
                label_embeddings = tf.boolean_mask(
                    full_embeddings,
                    tf.equal(full_labels, label))
                label_embeddings_mean = tf.reduce_mean(label_embeddings, axis=0)
                means.append(label_embeddings_mean)
                mean_distances.append(
                    float(tf.reduce_mean(
                        tf.reduce_sum(tf.square(label_embeddings - label_embeddings_mean), axis=1)
                    ))
                )
            closest_inter_distances = []
            for index, mean in enumerate(means):
                same_labels = tf.eye(num_labels)[index:index + 1]
                closest_inter_distances.append(float(
                    tf.reduce_min(compute_pairwise_distances(
                        mean[None],
                        tf.stack(means),
                        DistanceFunction.EUCLIDEAN_DISTANCE) + same_labels * 1e6, axis=1)
                ))
            within_mean_distances = stable_sqrt(tf.constant(mean_distances))
            self.cache['class_weights'] = tf.clip_by_value(
                within_mean_distances / tf.constant(closest_inter_distances),
                0.2,
                5
            )

        data = []
        for _ in range(batch_conf['num_batches'] * batch_conf.get('combine_batches', 1)):
            elements = self.get_next_batch(image_files, labels, batch_conf)
            data += elements

        return tf.data.Dataset.zip(
            self._create_datasets_from_elements(data, testing),
        ), len(data)

    def get_next_batch(self, image_files, labels, batch_conf):
        data = list(zip(image_files, labels))
        random.shuffle(data)

        batch_size = batch_conf['batch_size']
        if batch_conf.get('uniform'):
            return data[0:batch_size]
        group_size = batch_conf['group_size']
        num_groups = batch_size // group_size
        data_map = defaultdict(list)
        for image_file, label in data:
            data_map[label].append(image_file)

        data_map = dict(filter(lambda x: len(x[1]) >= group_size, data_map.items()))
        sampled_labels = np.random.choice(
            list(data_map.keys()), size=num_groups, replace=False)
        grouped_data = []
        for label in sampled_labels:
            for _ in range(group_size):
                image_file = data_map[label].pop()
                grouped_data.append((image_file, label))
        return grouped_data

    def get_raw_pairwise_distances(self, batch, model, distance_function, training=True):
        images, labels = batch
        embeddings = model(images, training=training)
        group_size = self.conf['batch_design']['group_size']
        pairwise_distances = compute_pairwise_distances(
            embeddings, embeddings, distance_function)
        matching_labels_matrix = pairwise_matching_matrix(labels, labels)
        weights = self.get_pairwise_weights(labels, group_size, model.extra_info)
        q_bias = self.conf['batch_design'].get('q_bias', 1.0)
        return (
            pairwise_distances,
            matching_labels_matrix,
            (1 / weights) ** q_bias,
        )

    def get_pairwise_distances(self, batch, model, distance_function, training=True):
        images, labels = batch
        embeddings = model(images, training=training)

        q_bias = self.conf['batch_design'].get('q_bias', 1.0)
        if self.conf['batch_design'].get('npair'):
            pairwise_distances, matching_labels_matrix = get_npair_distances(
                embeddings, self.conf['batch_design']['npair'], distance_function)
            weights = self.get_npair_pairwise_weights(
                labels, self.conf['batch_design']['npair'], model.extra_info)
            return (
                tf.reshape(pairwise_distances, [-1]),
                tf.reshape(matching_labels_matrix, [-1]),
                tf.reshape(1 / weights, [-1]) ** q_bias,
            )
        else:
            group_size = self.conf['batch_design']['group_size']
            pairwise_distances = compute_pairwise_distances(
                embeddings, embeddings, distance_function)
            matching_labels_matrix = pairwise_matching_matrix(labels, labels)
            weights = self.get_pairwise_weights(labels, group_size, model.extra_info)
            return (
                upper_triangular_part(pairwise_distances),
                upper_triangular_part(matching_labels_matrix),
                upper_triangular_part(1 / weights) ** q_bias,
            )

    @staticmethod
    def get_npair_pairwise_weights(labels, npair, extra_info):
        batch_size = int(labels.shape[0])
        group_size = 2
        num_groups = batch_size // group_size
        num_labels = extra_info['num_labels']
        evens = tf.range(num_groups, dtype=tf.int64) * 2
        even_labels = tf.gather(labels, evens)
        num_average_images_per_label = extra_info['num_images'] / extra_info['num_labels']
        label_counts = tf.gather(
            tf.constant(extra_info['label_counts'], dtype=tf.float32),
            even_labels) / num_average_images_per_label
        label_counts_multiplied = get_n_blocks(
            pairwise_product(label_counts, label_counts),
            npair)
        positive_label_counts = label_counts[:, None]
        negative_weights = (npair - 1) / (num_labels - 1) / label_counts_multiplied
        positive_weights = 1 / positive_label_counts / (positive_label_counts - 1 / num_average_images_per_label)
        matching_labels_matrix = get_n_blocks(tf.cast(tf.eye(num_groups), tf.bool), npair)
        weights = positive_weights * tf.cast(matching_labels_matrix, tf.float32) + negative_weights * tf.cast(~matching_labels_matrix, tf.float32)
        return weights

    def get_npair_weights(self, labels, npair, extra_info):
        batch_size = int(labels.shape[0])
        group_size = 2
        num_groups = batch_size // group_size
        evens = tf.range(num_groups, dtype=tf.int64) * 2
        even_labels = tf.gather(labels, evens)

        num_labels = extra_info['num_labels']
        log_uniform = npair * math.log(num_labels)
        weight_sum = float(sum(self.cache['class_weights']))
        label_weights = tf.gather(self.cache['class_weights'], tf.reshape(even_labels, [-1, npair]))
        log_label_weights = tf.reduce_sum(tf.log(label_weights) - math.log(weight_sum), axis=1)
        weights = tf.exp(log_uniform + log_label_weights)
        return tf.reshape(tf.transpose(tf.reshape(tf.tile(weights, [npair]), [-1, 2])), [-1])

    def get_pairwise_weights(self, labels, group_size, extra_info):
        batch_size = int(labels.shape[0])
        num_groups = batch_size // group_size
        num_images = extra_info['num_images']
        label_counts = tf.gather(
            tf.constant(extra_info['label_counts'], dtype=tf.float32),
            labels)
        positive_label_counts = label_counts
        matching_labels_matrix = pairwise_matching_matrix(labels, labels)
        label_counts_multiplied = pairwise_product(label_counts, label_counts)
        num_labels = extra_info['num_labels']

        if self.conf['batch_design'].get('negative_class_mining'):
            class_weights = tf.gather(self.cache['class_weights'], labels)
            class_weights = class_weights / sum(class_weights)
            class_weights_pairwise_sum = pairwise_sum(class_weights, class_weights)
            positive_weights = (group_size - 1) * num_images * (num_images - 1) / (
                    positive_label_counts * (positive_label_counts - 1) * num_labels * (batch_size - 1)) * (self.conf['loss']['l']) / (num_labels - 1) \
                               * (1 - tf.pow(1 - class_weights, num_groups))
            negative_weights = (num_groups - 1) * group_size * num_images * (num_images - 1) / (
                    (batch_size - 1) * num_labels * (num_labels - 1) * positive_label_counts * (positive_label_counts - 1)) \
                               * (1 - tf.pow(1 - class_weights, num_groups) - tf.pow(1 - class_weights[:, None], num_groups)
                                  + tf.pow(1 - class_weights_pairwise_sum, num_groups))
            weights = positive_weights * tf.cast(matching_labels_matrix, tf.float32) + negative_weights * tf.cast(~matching_labels_matrix, tf.float32)
            return weights
        if self.conf['loss'].get('new_importance_sampling'):
            positive_weights = (group_size - 1) * num_images * (num_images - 1) / (
                    positive_label_counts * (positive_label_counts - 1) * num_labels * (batch_size - 1)) * (self.conf['loss']['l']) / (num_labels - 1)
            negative_weights = (num_groups - 1) * group_size * num_images * (num_images - 1) / (
                    (batch_size - 1) * num_labels * (num_labels - 1) * label_counts_multiplied)
            weights = positive_weights * tf.cast(matching_labels_matrix, tf.float32) + negative_weights * tf.cast(~matching_labels_matrix, tf.float32)
            return weights
        if self.conf['loss'].get('balanced_pairs') and not self.conf['batch_design'].get('uniform'):
            positive_weights = 1.
            negative_weights = (num_groups - 1) * group_size / self.conf['loss']['l'] / (group_size - 1)
            weights = positive_weights * tf.cast(matching_labels_matrix, tf.float32) + negative_weights * tf.cast(~matching_labels_matrix, tf.float32)
            return weights
        if self.conf['loss'].get('balanced_pairs') and self.conf['batch_design'].get('uniform'):
            positive_weights =  (positive_label_counts * (positive_label_counts - 1) * num_labels) / \
                                (num_images * (num_images - 1))
            negative_weights = (num_labels * (num_labels - 1) * label_counts_multiplied) / \
                               (num_images * (num_images - 1) * self.conf['loss']['l'])
            weights = positive_weights * tf.cast(matching_labels_matrix, tf.float32) + negative_weights * tf.cast(~matching_labels_matrix, tf.float32)
            return weights
        positive_weights = (group_size - 1) * num_images * (num_images - 1) / (
                positive_label_counts * (positive_label_counts - 1) * num_labels * (batch_size - 1))
        negative_weights = (num_groups - 1) * group_size * num_images * (num_images - 1) / (
                (batch_size - 1) * num_labels * (num_labels - 1) * label_counts_multiplied)
        weights = positive_weights * tf.cast(matching_labels_matrix, tf.float32) + negative_weights * tf.cast(~matching_labels_matrix, tf.float32)
        return weights

    def get_npair_distances(self, batch, model, n, distance_function, training=True):
        if self.conf['batch_design']['group_size'] != 2:
            raise Exception('group size must be 2 in order to get npair distances')
        if (self.conf['batch_design']['batch_size'] // 2) % n != 0:
            raise Exception(
                'n does not divide the number of groups: n={}, num_groups={}'.format(
                    n, self.conf['batch_size'] // 2
                ))

        images, labels = batch
        embeddings = model(images, training=training)

        return get_npair_distances(embeddings, n, distance_function)

    def get_embeddings(self, batch, model, distance_function, training=True):
        images, _ = batch
        return model(images, training=training)
