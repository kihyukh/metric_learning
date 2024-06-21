from util.registry.batch_design import BatchDesign

from collections import defaultdict
from util.tensor_operations import compute_elementwise_distances

import numpy as np
import tensorflow as tf
import random


class PairBatchDesign(BatchDesign):
    name = 'pair'

    def get_next_batch(self, image_files, labels, batch_conf):
        data_map = defaultdict(list)
        data_list = list(zip(image_files, labels))
        random.shuffle(data_list)
        for image_file, label in data_list:
            data_map[label].append(image_file)
        data_map = dict(filter(lambda x: len(x[1]) >= 2, data_map.items()))

        batch_size = batch_conf['batch_size']
        positive_ratio = batch_conf['positive_ratio']
        label_match = [1 if random.random() < positive_ratio else 0
                       for _ in range(batch_size // 2)]

        elements = []
        for match in label_match:
            if match:
                query_label = np.random.choice(list(data_map.keys()), size=1)[0]
                a = data_map[query_label].pop()
                b = data_map[query_label].pop()
                elements.append((a, query_label))
                elements.append((b, query_label))
                if len(data_map[query_label]) < 2:
                    del data_map[query_label]
            else:
                query_label, target_label = np.random.choice(
                    list(data_map.keys()), size=2, replace=False)
                elements.append(
                    (data_map[query_label].pop(), query_label)
                )
                elements.append(
                    (data_map[target_label].pop(), target_label)
                )
                if len(data_map[query_label]) < 2:
                    del data_map[query_label]
                if len(data_map[target_label]) < 2:
                    del data_map[target_label]
        return elements

    def get_pairwise_distances(self, batch, model, distance_function, training=True):
        images, labels = batch
        embeddings = model(images, training=training)
        evens = tf.range(images.shape[0] // 2, dtype=tf.int64) * 2
        odds = tf.range(images.shape[0] // 2, dtype=tf.int64) * 2 + 1
        even_embeddings = tf.gather(embeddings, evens)
        odd_embeddings = tf.gather(embeddings, odds)
        even_labels = tf.gather(labels, evens)
        odd_labels = tf.gather(labels, odds)
        match = tf.equal(even_labels, odd_labels)

        elementwise_distances = compute_elementwise_distances(
            even_embeddings, odd_embeddings, distance_function
        )

        weights = self.get_pairwise_weights(
            labels,
            self.conf['batch_design']['positive_ratio'],
            model.extra_info)
        q_bias = self.conf['batch_design'].get('q_bias', 1.0)
        return elementwise_distances, match, (1 / weights) ** q_bias

    def get_pairwise_weights(self, labels, positive_ratio, extra_info):
        num_images = extra_info['num_images']
        num_labels = extra_info['num_labels']
        label_counts = tf.gather(
            tf.constant(extra_info['label_counts'], dtype=tf.float32),
            labels)
        evens = tf.range(labels.shape[0] // 2, dtype=tf.int64) * 2
        odds = tf.range(labels.shape[0] // 2, dtype=tf.int64) * 2 + 1
        even_labels = tf.gather(labels, evens)
        odd_labels = tf.gather(labels, odds)
        match = tf.equal(even_labels, odd_labels)
        if self.conf['loss'].get('balanced_pairs'):
            positive_weights = positive_ratio
            negative_weights = (1 - positive_ratio) / self.conf['loss']['l']
            weights = positive_weights * tf.cast(match, tf.float32) + negative_weights * tf.cast(~match, tf.float32)
            return weights
        even_label_counts = tf.gather(label_counts, evens)
        odd_label_counts = tf.gather(label_counts, odds)
        label_counts_multiplied = tf.multiply(even_label_counts, odd_label_counts)
        positive_weights = ((positive_ratio * num_images * (num_images - 1)) /
                            (even_label_counts * (even_label_counts - 1) * num_labels))
        negative_weights = ((1 - positive_ratio) * num_images * (num_images - 1)) /\
                           (num_labels * (num_labels - 1) * label_counts_multiplied)
        weights = positive_weights * tf.cast(match, tf.float32) + negative_weights * tf.cast(~match, tf.float32)
        return weights

    def get_npair_distances(self, batch, model, n, distance_function, training=True):
        raise NotImplementedError

    def get_embeddings(self, batch, model, distance_function, training=True):
        raise NotImplementedError
