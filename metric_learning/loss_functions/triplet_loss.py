import tensorflow as tf

from util.registry.loss_function import LossFunction
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import upper_triangular_part
from util.tensor_operations import repeat_columns
from util.tensor_operations import pairwise_difference
from metric_learning.constants.distance_function import DistanceFunction


def masked_maximum(data, mask, dim=1):
    axis_minimums = tf.reduce_min(data, dim, keepdims=True)
    masked_maximums = tf.reduce_max(
        tf.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    axis_maximums = tf.reduce_max(data, dim, keepdims=True)
    masked_minimums = tf.reduce_min(
        tf.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums


class TripletLossFunction(LossFunction):
    name = 'triplet'

    def loss(self, batch, model, dataset):
        images, labels = batch
        pdist_matrix, adjacency, weights = dataset.get_raw_pairwise_distances(
            batch, model, DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED)

        lshape = tf.shape(labels)
        labels = tf.reshape(labels, [lshape[0], 1])

        adjacency_not = tf.logical_not(adjacency)

        batch_size = tf.size(labels)

        # Compute the mask.
        pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
        mask = tf.logical_and(
            tf.tile(adjacency_not, [batch_size, 1]),
            tf.greater(
                pdist_matrix_tile, tf.reshape(
                    tf.transpose(pdist_matrix), [-1, 1])))
        mask_final = tf.reshape(
            tf.greater(
                tf.reduce_sum(
                    tf.cast(mask, dtype=tf.float32), 1, keepdims=True),
                0.0), [batch_size, batch_size])
        mask_final = tf.transpose(mask_final)

        adjacency_not = tf.cast(adjacency_not, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = tf.reshape(
            masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
        negatives_outside = tf.transpose(negatives_outside)

        # negatives_inside: largest D_an.
        negatives_inside = tf.tile(
            masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
        semi_hard_negatives = tf.where(
            mask_final, negatives_outside, negatives_inside)

        margin = self.conf['loss']['alpha']
        loss_mat = tf.add(margin, pdist_matrix - semi_hard_negatives)

        mask_positives = tf.cast(
            adjacency, dtype=tf.float32) - tf.diag(
            tf.ones([batch_size]))

        num_positives = tf.reduce_sum(mask_positives)

        triplet_loss = tf.truediv(
            tf.reduce_sum(
                tf.maximum(
                    tf.multiply(loss_mat, mask_positives), 0.0)),
            num_positives)

        return triplet_loss
