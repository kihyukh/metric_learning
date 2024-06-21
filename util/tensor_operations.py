import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction


def compute_pairwise_distances(first, second, distance_function):
    if distance_function == DistanceFunction.COSINE_SIMILARITY:
        first_norm = first / tf.norm(first, axis=1, keep_dims=True)
        second_norm = second / tf.norm(second, axis=1, keep_dims=True)
        return -tf.reduce_sum(
            tf.multiply(second_norm[None], first_norm[:, None]),
            axis=2)
    if distance_function == DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED:
        return tf.reduce_sum(
            tf.square(second[None] - first[:, None]),
            axis=2)
    if distance_function == DistanceFunction.EUCLIDEAN_DISTANCE:
        return stable_sqrt(tf.reduce_sum(
            tf.square(second[None] - first[:, None]),
            axis=2))
    if distance_function == DistanceFunction.DOT_PRODUCT:
        return -tf.reduce_sum(
            tf.multiply(second[None], first[:, None]),
            axis=2)
    raise Exception(
        'Unknown distance function with name {}'.format(distance_function))


def compute_elementwise_distances(first, second, distance_function):
    if distance_function == DistanceFunction.COSINE_SIMILARITY:
        first_norm = tf.norm(first, axis=1)
        second_norm = tf.norm(second, axis=1)
        return -tf.reduce_sum(tf.multiply(first, second), axis=1) / first_norm / second_norm
    if distance_function == DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED:
        return tf.reduce_sum(tf.square(first - second), axis=1)
    if distance_function == DistanceFunction.EUCLIDEAN_DISTANCE:
        return stable_sqrt(tf.reduce_sum(tf.square(first - second), axis=1))
    if distance_function == DistanceFunction.DOT_PRODUCT:
        return -tf.reduce_sum(tf.square(first - second), axis=1)
    raise Exception(
        'Unknown distance function with name {}'.format(distance_function))


def pairwise_difference(first, second):
    return -second[None] + first[:, None]


def pairwise_matching_matrix(first, second):
    return tf.equal(second[None], first[:, None])


def repeat_columns(labels):
    return tf.tile(labels[:, None], [1, labels.shape[0]])


def upper_triangular_part(matrix):
    a = tf.linalg.band_part(tf.ones(matrix.shape), -1, 0)
    return tf.boolean_mask(matrix, 1 - a)


def off_diagonal_part(matrix):
    return tf.boolean_mask(matrix, 1 - tf.eye(int(matrix.shape[0])))


def stable_sqrt(tensor):
    return tf.sqrt(tf.maximum(tensor, 1e-12))


def get_n_blocks(tensor, n, transpose=False):
    r = tf.range(tensor.shape[0])
    mask = tf.equal(r[None] // n, r[:, None] // n)
    if transpose:
        return tf.transpose(
            tf.reshape(tf.boolean_mask(tf.transpose(tensor), mask), [-1, n])
        )
    return tf.reshape(tf.boolean_mask(tensor, mask), [-1, n])


def pairwise_product(first, second):
    return tf.multiply(second[None], first[:, None])


def pairwise_sum(first, second):
    return second[None] + first[:, None]
