import tensorflow as tf

import numpy as np

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.loss_function import LossFunction

from util.tensor_operations import off_diagonal_part
from util.tensor_operations import repeat_columns

tfe = tf.contrib.eager


def distance_weight(distances, dimension):
    log_weights = (-(dimension - 2) * tf.log(distances) -
                   (dimension - 3) / 2 * tf.log((1 - tf.square(distances) / 4)))
    weights = tf.exp(log_weights - max(log_weights))
    return weights / sum(weights)


class MarginLoss(LossFunction):
    name = 'margin'

    def __init__(self, conf, extra_info):
        super(MarginLoss, self).__init__(conf, extra_info)

        loss_conf = conf['loss']
        if 'num_labels' in extra_info and 'num_images' in extra_info:
            beta_class = tf.ones(extra_info['num_labels']) * loss_conf['beta']

            self.extra_variables['beta'] = tfe.Variable(beta_class)

    def loss(self, batch, model, dataset):
        images, labels = batch
        pairwise_distances, matching_labels_matrix, weights = dataset.get_raw_pairwise_distances(
            batch, model, DistanceFunction.EUCLIDEAN_DISTANCE
        )

        # distance weighted sampling
        group_size = self.conf['batch_design']['group_size']
        negative_distance_list = []
        positive_distance_list = []

        for i in range(int(pairwise_distances.shape[0])):
            distances = pairwise_distances[i, :]
            distances = tf.maximum(distances, 0.5)
            match = matching_labels_matrix[i, :]
            positive_match = (matching_labels_matrix & ~tf.cast(tf.eye(int(pairwise_distances.shape[0])), tf.bool))[i, :]
            nonzero_loss_cutoff = (distances < self.conf['loss']['beta'] + self.conf['loss']['alpha'])
            negatives = tf.boolean_mask(distances, ~match)
            if int(negatives.shape[0]) > 0:
                weights = distance_weight(negatives, self.conf['model']['dimension']).numpy()
                negative_distance_list.append(np.random.choice(negatives, size=group_size - 1, p=weights))
                positive_distance_list.append(tf.boolean_mask(distances, positive_match))
            else:
                negative_distance_list.append(tf.ones(group_size - 1) * 100)
                positive_distance_list.append(tf.zeros(group_size - 1))
        negative_distances = tf.stack(negative_distance_list)
        positive_distances = tf.stack(positive_distance_list)

        betas = tf.gather(self.extra_variables['beta'], labels)[:, None]

        alpha = self.conf['loss']['alpha']
        positive_loss = tf.maximum(positive_distances - betas + alpha, 0.0)
        negative_loss = tf.maximum(betas - negative_distances + alpha, 0.0)

        pair_cnt = tf.reduce_sum(tf.cast(positive_loss > 0, tf.float32)) + tf.reduce_sum(tf.cast(negative_loss > 0, tf.float32))

        nu = self.conf['loss']['nu']

        return (tf.reduce_sum(positive_loss + negative_loss) + nu * tf.reduce_sum(betas)) / pair_cnt
