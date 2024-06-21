import tensorflow as tf

from util.distribution import factor_expansion
from util.distribution import wallenius

tf.enable_eager_execution()


class TensorOperationsTest(tf.test.TestCase):
    def testFactorExpansion(self):
        exponents = tf.constant([[1, 2], [0, 0]])
        terms, signs = factor_expansion(exponents)
        self.assertAllEqual(terms, [
            [0., 2, 1, 3],
            [0, 0, 0, 0],
        ])
        self.assertAllEqual(signs, [
            [1., -1, -1, 1],
            [1, -1, -1, 1],
        ])

    def testWallenius(self):
        weights = tf.constant([1, 1, 1, 1, 1, 1], dtype=tf.float32)
        self.assertAllClose(wallenius(tf.constant([[0]]), weights), [1 / 6])
        self.assertAllClose(wallenius(tf.constant([[0, 1]]), weights), [1 / 15])
        self.assertAllClose(wallenius(tf.constant([[0, 1, 2]]), weights), [1 / 20])
        self.assertAllClose(wallenius(tf.constant([[0, 1, 2, 3]]), weights), [1 / 15])
        self.assertAllClose(wallenius(tf.constant([[0, 1, 2, 3, 4]]), weights), [1 / 6])


if __name__ == '__main__':
    tf.test.main()
