import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction

from util.registry.loss_function import LossFunction
from util.tensor_operations import compute_pairwise_distances
from util.tensor_operations import pairwise_matching_matrix


class LiftedStructureLoss(LossFunction):
    name = 'lifted'

    def loss(self, batch, model, dataset):
        loss_conf = self.conf['loss']
        embeddings = dataset.get_embeddings(batch, model, None)
        pairwise_distances = compute_pairwise_distances(
            embeddings, embeddings, DistanceFunction.EUCLIDEAN_DISTANCE)
        _, labels = batch
        adjacency = pairwise_matching_matrix(labels, labels)

        diff = loss_conf['alpha'] - pairwise_distances
        mask = tf.cast(~adjacency, dtype=tf.float32)
        row_minimums = tf.reduce_min(diff, 1, keepdims=True)
        row_negative_maximums = tf.reduce_max(
            tf.multiply(diff - row_minimums, mask), 1,
            keepdims=True) + row_minimums

        max_elements = tf.maximum(
            row_negative_maximums, tf.transpose(row_negative_maximums))
        batch_size = int(labels.shape[0])
        diff_tiled = tf.tile(diff, [batch_size, 1])
        mask_tiled = tf.tile(mask, [batch_size, 1])
        max_elements_vect = tf.reshape(tf.transpose(max_elements), [-1, 1])

        loss_exp_left = tf.reshape(
            tf.reduce_sum(
                tf.multiply(
                    tf.exp(diff_tiled - max_elements_vect), mask_tiled),
                1,
                keepdims=True), [batch_size, batch_size])
        loss_mat = max_elements + tf.log(
            loss_exp_left + tf.transpose(loss_exp_left))
        # Add the positive distance.
        loss_mat += pairwise_distances

        mask_positives = tf.cast(
            adjacency, dtype=tf.float32) - tf.diag(
            tf.ones([batch_size]))

        num_positives = tf.reduce_sum(mask_positives) / 2.0

        lifted_loss = tf.truediv(
            0.25 * tf.reduce_sum(
                tf.square(
                    tf.maximum(
                        tf.multiply(loss_mat, mask_positives), 0.0))),
            num_positives)
        return lifted_loss
