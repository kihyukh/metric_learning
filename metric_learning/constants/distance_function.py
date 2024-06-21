from enum import Enum


class DistanceFunction(Enum):
    EUCLIDEAN_DISTANCE = 1
    EUCLIDEAN_DISTANCE_SQUARED = 2
    DOT_PRODUCT = 3
    COSINE_SIMILARITY = 4


def get_distance_function(name):
    if name == 'dot_product':
        return DistanceFunction.DOT_PRODUCT
    if name == 'euclidean_distance':
        return DistanceFunction.EUCLIDEAN_DISTANCE
    if name == 'euclidean_distance_squared':
        return DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED
    if name == 'cosine_similarity':
        return DistanceFunction.COSINE_SIMILARITY
    raise Exception('Unknown distance function with name {}'.format(name))
