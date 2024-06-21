import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction

from util.registry.loss_function import LossFunction
from util.tensor_operations import compute_pairwise_distances
from util.tensor_operations import pairwise_matching_matrix

from sklearn import metrics


def get_cluster_assignment(pairwise_distances, centroid_ids):
    predictions = tf.argmin(
        tf.gather(pairwise_distances, centroid_ids), dimension=0)
    batch_size = tf.shape(pairwise_distances)[0]

    # Deal with numerical instability
    mask = tf.reduce_any(tf.one_hot(
        centroid_ids, batch_size, True, False, axis=-1, dtype=tf.bool),
        axis=0)
    constraint_one_hot = tf.multiply(
        tf.one_hot(centroid_ids,
                   batch_size,
                   tf.constant(1, dtype=tf.int64),
                   tf.constant(0, dtype=tf.int64),
                   axis=0,
                   dtype=tf.int64),
        tf.to_int64(tf.range(tf.shape(centroid_ids)[0])))
    constraint_vect = tf.reduce_sum(tf.transpose(constraint_one_hot), axis=0)

    y_fixed = tf.where(mask, constraint_vect, predictions)
    return y_fixed


def update_1d_tensor(y, index, value):
    value = tf.squeeze(value)
    y_before = tf.slice(y, [0], [index])
    y_after = tf.slice(y, [index + 1], [-1])
    y_mod = tf.concat([y_before, [value], y_after], 0)
    return y_mod


def _compute_nmi_score(labels, predictions):
    return tf.to_float(
        tf.py_func(
            metrics.normalized_mutual_info_score, [labels, predictions],
            [tf.float64],
            name='nmi'))


def _compute_ami_score(labels, predictions):
    ami_score = tf.to_float(
        tf.py_func(
            metrics.adjusted_mutual_info_score, [labels, predictions],
            [tf.float64],
            name='ami'))
    return tf.maximum(0.0, ami_score)


def _compute_ari_score(labels, predictions):
    ari_score = tf.to_float(
        tf.py_func(
            metrics.adjusted_rand_score, [labels, predictions], [tf.float64],
            name='ari'))
    return tf.maximum(0.0, ari_score)


def _compute_vmeasure_score(labels, predictions):
    vmeasure_score = tf.to_float(
        tf.py_func(
            metrics.v_measure_score, [labels, predictions], [tf.float64],
            name='vmeasure'))
    return tf.maximum(0.0, vmeasure_score)


def _compute_zeroone_score(labels, predictions):
    zeroone_score = tf.to_float(
        tf.equal(
            tf.reduce_sum(
                tf.to_int32(tf.equal(labels, predictions))),
            tf.shape(labels)[0]))
    return zeroone_score


def compute_clustering_score(labels, predictions, margin_type):
    """Computes the clustering score via sklearn.metrics functions.
    There are various ways to compute the clustering score. Intuitively,
    we want to measure the agreement of two clustering assignments (labels vs
    predictions) ignoring the permutations and output a score from zero to one.
    (where the values close to one indicate significant agreement).
    This code supports following scoring functions:
      nmi: normalized mutual information
      ami: adjusted mutual information
      ari: adjusted random index
      vmeasure: v-measure
      const: indicator checking whether the two clusterings are the same.
    See http://scikit-learn.org/stable/modules/classes.html#clustering-metrics
      for the detailed descriptions.
    Args:
      labels: 1-D Tensor. ground truth cluster assignment.
      predictions: 1-D Tensor. predicted cluster assignment.
      margin_type: Type of structured margin to use. Default is nmi.
    Returns:
      clustering_score: dtypes.float32 scalar.
        The possible valid values are from zero to one.
        Zero means the worst clustering and one means the perfect clustering.
    Raises:
      ValueError: margin_type is not recognized.
    """
    margin_type_to_func = {
        'nmi': _compute_nmi_score,
        'ami': _compute_ami_score,
        'ari': _compute_ari_score,
        'vmeasure': _compute_vmeasure_score,
        'const': _compute_zeroone_score
    }

    if margin_type not in margin_type_to_func:
        raise ValueError('Unrecognized margin_type: %s' % margin_type)
    clustering_score_fn = margin_type_to_func[margin_type]
    return tf.squeeze(clustering_score_fn(labels, predictions))


def update_medoid_per_cluster(pairwise_distances, pairwise_distances_subset,
                              labels, chosen_ids, cluster_member_ids,
                              cluster_idx, margin_multiplier, margin_type):
    """Updates the cluster medoid per cluster.
    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      pairwise_distances_subset: 2-D Tensor of pairwise distances for one cluster.
      labels: 1-D Tensor of ground truth cluster assignment.
      chosen_ids: 1-D Tensor of cluster centroid indices.
      cluster_member_ids: 1-D Tensor of cluster member indices for one cluster.
      cluster_idx: Index of this one cluster.
      margin_multiplier: multiplication constant.
      margin_type: Type of structured margin to use. Default is nmi.
    Returns:
      chosen_ids: Updated 1-D Tensor of cluster centroid indices.
    """

    def func_cond(iteration, scores_margin):
        del scores_margin  # Unused variable scores_margin.
        return iteration < num_candidates

    def func_body(iteration, scores_margin):
        # swap the current medoid with the candidate cluster member
        candidate_medoid = tf.to_int32(cluster_member_ids[iteration])
        tmp_chosen_ids = update_1d_tensor(chosen_ids, cluster_idx, candidate_medoid)
        predictions = get_cluster_assignment(pairwise_distances, tmp_chosen_ids)
        metric_score = compute_clustering_score(labels, predictions, margin_type)
        pad_before = tf.zeros([iteration])
        pad_after = tf.zeros([num_candidates - 1 - iteration])
        return iteration + 1, scores_margin + tf.concat(
            [pad_before, [1.0 - metric_score], pad_after], 0)

    # pairwise_distances_subset is of size [p, 1, 1, p],
    #   the intermediate dummy dimensions at
    #   [1, 2] makes this code work in the edge case where p=1.
    #   this happens if the cluster size is one.
    scores_fac = -1.0 * tf.reduce_sum(
        tf.squeeze(pairwise_distances_subset, [1, 2]), axis=0)

    iteration = tf.constant(0)
    num_candidates = tf.size(cluster_member_ids)
    scores_margin = tf.zeros([num_candidates])

    _, scores_margin = tf.while_loop(func_cond, func_body,
                                                   [iteration, scores_margin])
    candidate_scores = tf.add(scores_fac, margin_multiplier * scores_margin)

    argmax_index = tf.to_int32(tf.argmax(candidate_scores, axis=0))

    best_medoid = tf.to_int32(cluster_member_ids[argmax_index])
    chosen_ids = update_1d_tensor(chosen_ids, cluster_idx, best_medoid)
    return chosen_ids


def update_all_medoids(pairwise_distances, predictions, labels, chosen_ids,
                       margin_multiplier, margin_type):
    def func_cond_augmented_pam(iteration, chosen_ids):
        del chosen_ids  # Unused argument.
        return iteration < num_classes

    def func_body_augmented_pam(iteration, chosen_ids):
        """Call the update_medoid_per_cluster subroutine."""
        mask = tf.equal(
            tf.to_int64(predictions), tf.to_int64(iteration))
        this_cluster_ids = tf.where(mask)

        pairwise_distances_subset = tf.transpose(
            tf.gather(
                tf.transpose(
                    tf.gather(pairwise_distances, this_cluster_ids)),
                this_cluster_ids))

        chosen_ids = update_medoid_per_cluster(pairwise_distances,
                                               pairwise_distances_subset, labels,
                                               chosen_ids, this_cluster_ids,
                                               iteration, margin_multiplier,
                                               margin_type)
        return iteration + 1, chosen_ids

    unique_class_ids = tf.unique(labels)[0]
    num_classes = tf.size(unique_class_ids)
    iteration = tf.constant(0)

    _, chosen_ids = tf.while_loop(
        func_cond_augmented_pam, func_body_augmented_pam, [iteration, chosen_ids])
    return chosen_ids


def _find_loss_augmented_facility_idx(pairwise_distances, labels, chosen_ids,
                                      candidate_ids, margin_multiplier,
                                      margin_type):
    """Find the next centroid that maximizes the loss augmented inference.
    This function is a subroutine called from compute_augmented_facility_locations
    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      labels: 1-D Tensor of ground truth cluster assignment.
      chosen_ids: 1-D Tensor of current centroid indices.
      candidate_ids: 1-D Tensor of candidate indices.
      margin_multiplier: multiplication constant.
      margin_type: Type of structured margin to use. Default is nmi.
    Returns:
      integer index.
    """
    num_candidates = tf.shape(candidate_ids)[0]

    pairwise_distances_chosen = tf.gather(pairwise_distances, chosen_ids)
    pairwise_distances_candidate = tf.gather(
        pairwise_distances, candidate_ids)
    pairwise_distances_chosen_tile = tf.tile(
        pairwise_distances_chosen, [1, num_candidates])

    candidate_scores = -1.0 * tf.reduce_sum(
        tf.reshape(
            tf.reduce_min(
                tf.concat([
                    pairwise_distances_chosen_tile,
                    tf.reshape(pairwise_distances_candidate, [1, -1])
                ], 0),
                axis=0,
                keepdims=True), [num_candidates, -1]),
        axis=1)

    nmi_scores = tf.zeros([num_candidates])
    iteration = tf.constant(0)

    def func_cond(iteration, nmi_scores):
        del nmi_scores  # Unused in func_cond()
        return iteration < num_candidates

    def func_body(iteration, nmi_scores):
        predictions = get_cluster_assignment(
            pairwise_distances,
            tf.concat([chosen_ids, [candidate_ids[iteration]]], 0))
        nmi_score_i = compute_clustering_score(labels, predictions, margin_type)
        pad_before = tf.zeros([iteration])
        pad_after = tf.zeros([num_candidates - 1 - iteration])
        # return 1 - NMI score as the structured loss.
        #   because NMI is higher the better [0,1].
        return iteration + 1, nmi_scores + tf.concat(
            [pad_before, [1.0 - nmi_score_i], pad_after], 0)

    _, nmi_scores = tf.while_loop(
        func_cond, func_body, [iteration, nmi_scores])

    candidate_scores = tf.add(
        candidate_scores, margin_multiplier * nmi_scores)

    argmax_index = tf.to_int32(
        tf.argmax(candidate_scores, axis=0))

    return candidate_ids[argmax_index]


def compute_augmented_facility_locations(pairwise_distances, labels, all_ids,
                                         margin_multiplier, margin_type):
    """Computes the centroid locations.
    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      labels: 1-D Tensor of ground truth cluster assignment.
      all_ids: 1-D Tensor of all data indices.
      margin_multiplier: multiplication constant.
      margin_type: Type of structured margin to use. Default is nmi.
    Returns:
      chosen_ids: 1-D Tensor of chosen centroid indices.
    """

    def func_cond_augmented(iteration, chosen_ids):
        del chosen_ids  # Unused argument in func_cond_augmented.
        return iteration < num_classes

    def func_body_augmented(iteration, chosen_ids):
        # find a new facility location to add
        #  based on the clustering score and the NMI score
        candidate_ids = tf.setdiff1d(all_ids, chosen_ids)[0]
        new_chosen_idx = _find_loss_augmented_facility_idx(pairwise_distances,
                                                           labels, chosen_ids,
                                                           candidate_ids,
                                                           margin_multiplier,
                                                           margin_type)
        chosen_ids = tf.concat([chosen_ids, [new_chosen_idx]], 0)
        return iteration + 1, chosen_ids

    num_classes = tf.size(tf.unique(labels)[0])
    chosen_ids = tf.constant(0, dtype=tf.int32, shape=[0])

    # num_classes get determined at run time based on the sampled batch.
    iteration = tf.constant(0)

    _, chosen_ids = tf.while_loop(
        func_cond_augmented,
        func_body_augmented, [iteration, chosen_ids],
        shape_invariants=[iteration.get_shape(), tf.TensorShape(
            [None])])
    return chosen_ids


def compute_facility_energy(pairwise_distances, centroid_ids):
    return -1.0 * tf.reduce_sum(
        tf.reduce_min(
            tf.gather(pairwise_distances, centroid_ids), axis=0))


def compute_augmented_facility_locations_pam(pairwise_distances,
                                             labels,
                                             margin_multiplier,
                                             margin_type,
                                             chosen_ids,
                                             pam_max_iter=5):
    for _ in range(pam_max_iter):
        predictions = get_cluster_assignment(pairwise_distances, chosen_ids)

        chosen_ids = update_all_medoids(pairwise_distances, predictions, labels,
                                        chosen_ids, margin_multiplier, margin_type)
    return chosen_ids


def compute_gt_cluster_score(pairwise_distances, labels):
    """Compute ground truth facility location score.
    Loop over each unique classes and compute average travel distances.
    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      labels: 1-D Tensor of ground truth cluster assignment.
    Returns:
      gt_cluster_score: dtypes.float32 score.
    """
    unique_class_ids = tf.unique(labels)[0]
    num_classes = tf.size(unique_class_ids)
    iteration = tf.constant(0)
    gt_cluster_score = tf.constant(0.0, dtype=tf.float32)

    def func_cond(iteration, gt_cluster_score):
        del gt_cluster_score  # Unused argument.
        return iteration < num_classes

    def func_body(iteration, gt_cluster_score):
        """Per each cluster, compute the average travel distance."""
        mask = tf.equal(labels, unique_class_ids[iteration])
        this_cluster_ids = tf.where(mask)
        pairwise_distances_subset = tf.transpose(
            tf.gather(
                tf.transpose(
                    tf.gather(pairwise_distances, this_cluster_ids)),
                this_cluster_ids))
        this_cluster_score = -1.0 * tf.reduce_min(
            tf.reduce_sum(
                pairwise_distances_subset, axis=0))
        return iteration + 1, gt_cluster_score + this_cluster_score

    _, gt_cluster_score = tf.while_loop(
        func_cond, func_body, [iteration, gt_cluster_score])
    return gt_cluster_score


class ClusterLoss(LossFunction):
    name = 'cluster'

    def loss(self, batch, model, dataset):
        margin_multiplier = self.conf['loss']['gamma']
        margin_type = 'nmi'

        embeddings = dataset.get_embeddings(batch, model, None)
        pairwise_distances = compute_pairwise_distances(
            embeddings, embeddings, DistanceFunction.EUCLIDEAN_DISTANCE)
        _, labels = batch
        all_ids = tf.range(tf.shape(embeddings)[0])

        chosen_ids = compute_augmented_facility_locations(pairwise_distances, labels,
                                                          all_ids, margin_multiplier,
                                                          margin_type)
        chosen_ids = compute_augmented_facility_locations_pam(pairwise_distances,
                                                              labels,
                                                              margin_multiplier,
                                                              margin_type,
                                                              chosen_ids)
        score_pred = compute_facility_energy(pairwise_distances, chosen_ids)

        predictions = get_cluster_assignment(pairwise_distances, chosen_ids)
        clustering_score_pred = compute_clustering_score(labels, predictions,
                                                         margin_type)
        score_gt = compute_gt_cluster_score(pairwise_distances, labels)

        clustering_loss = tf.maximum(
            score_pred + margin_multiplier * (1.0 - clustering_score_pred) - score_gt,
            0.0)
        clustering_loss.set_shape([])

        return clustering_loss
