from util.registry.metric import Metric

from collections import defaultdict

import tensorflow as tf

import sklearn.cluster
import sklearn.metrics.cluster


class NMI(Metric):
    name = 'nmi'

    def compute_metric(self, model, ds, num_testcases):
        embeddings_list, labels_list = self.get_embeddings(
            model, ds, num_testcases)
        embeddings = tf.concat(embeddings_list, axis=0)
        labels = tf.concat(labels_list, axis=0)
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[int(label)] += 1
        clusters = sklearn.cluster.KMeans(len(label_counts)).fit(embeddings).labels_
        return sklearn.metrics.cluster.normalized_mutual_info_score(clusters, labels)
