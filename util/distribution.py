import tensorflow as tf

from itertools import product
from functools import reduce


def factor_expansion(exponents):
    ret = []
    for exponent_list in exponents:
        one_row = []
        for expanded in product(*[[0, exponent] for exponent in exponent_list]):
            one_row.append(float(sum(expanded)))
        ret.append(tf.constant(one_row))
    signs = []
    for exponent_list in exponents:
        one_row = []
        for expanded in product(*[[1, -1] for _ in exponent_list]):
            one_row.append(float(reduce(lambda x, y: x * y, expanded)))
        signs.append(tf.constant(one_row))
    return tf.stack(ret), tf.stack(signs)


def integrate(exponents, signs):
    return tf.reduce_sum(signs / (exponents + 1), axis=1)


def wallenius(x, w):
    w_chosen = tf.gather(w, x)
    d = sum(w) - tf.reduce_sum(w_chosen, axis=1)
    return integrate(*factor_expansion(w_chosen / d[:, None]))
