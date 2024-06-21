import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.loss_function import LossFunction


class L1ContrastiveLossFunction(LossFunction):
    name = 'l1_contrastive'

    def loss(self, batch, model, dataset):
        alpha = self.conf['loss']['alpha']

        pairwise_distances, matching_labels_matrix, weights = dataset.get_pairwise_distances(
            batch, model, DistanceFunction.EUCLIDEAN_DISTANCE)
        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)
        if self.conf['loss'].get('importance_sampling') or self.conf['loss'].get('new_importance_sampling') or self.conf['loss'].get('balanced_pairs'):
            positive_weights = tf.boolean_mask(weights, matching_labels_matrix)
            negative_weights = tf.boolean_mask(weights, ~matching_labels_matrix)
            loss_value = (
                sum(positive_distances * positive_weights) +
                sum(tf.maximum(0, alpha - negative_distances) * negative_weights)
            ) / int(pairwise_distances.shape[0])
        else:
            loss_value = (
                sum(positive_distances) +
                sum(tf.maximum(0, alpha - negative_distances))
            ) / int(pairwise_distances.shape[0])

        return loss_value
