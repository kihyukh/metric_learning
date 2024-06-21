import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.loss_function import LossFunction
from metric_learning.batch_designs.grouped import get_npair_distances


class NPairLossFunction(LossFunction):
    name = 'npair'

    def loss(self, batch, model, dataset):
        embeddings = dataset.get_embeddings(
            batch, model, DistanceFunction.DOT_PRODUCT)
        pairwise_distances, matching_matrix = get_npair_distances(
            embeddings, self.conf['batch_design']['npair'],
            DistanceFunction.DOT_PRODUCT, transpose=False)
        pairwise_distances_t, matching_matrix_t = get_npair_distances(
            embeddings, self.conf['batch_design']['npair'],
            DistanceFunction.DOT_PRODUCT, transpose=True)
        regularizer = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings), axis=1))

        if self.conf['loss'].get('importance_sampling') and self.conf['batch_design'].get('negative_class_mining'):
            _, labels = batch
            weights = dataset.get_npair_weights(
                labels, self.conf['batch_design']['npair'], model.extra_info)
            return tf.reduce_mean(
                (0.5 * tf.reduce_logsumexp(-pairwise_distances, axis=1) +
                0.5 * tf.reduce_logsumexp(-pairwise_distances_t, axis=0) +
                tf.boolean_mask(pairwise_distances, matching_matrix))
            / weights) + regularizer * self.conf['loss']['lambda']
        else:
            return tf.reduce_mean(
                0.5 * tf.reduce_logsumexp(-pairwise_distances, axis=1) +
                0.5 * tf.reduce_logsumexp(-pairwise_distances_t, axis=0) +
                tf.boolean_mask(pairwise_distances, matching_matrix)
            ) + regularizer * self.conf['loss']['lambda']
