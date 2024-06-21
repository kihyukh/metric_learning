import tensorflow as tf

from metric_learning.metrics.accuracy import evaluate_accuracy
from metric_learning.metrics.accuracy import euclidean_distance


tf.enable_eager_execution()


class AccuracyMetricTest(tf.test.TestCase):
    def testAccuracy(self):
        anchor_embeddings = tf.constant([
            [0., 0.],
            [0., 0.],
            [0., 0.],
        ])
        positive_embeddings = tf.constant([
            [0., 1.],
            [0., 5.],
            [1., 1.],
        ])
        negative_embeddings = tf.stack([
            tf.constant([[0., 2.], [0., 2.], [2., 0.]]),
            tf.constant([[2., 0.], [2., 0.], [2., -2.]]),
            tf.constant([[0., -2.], [0., 0.], [2., 1.]]),
            tf.constant([[-1.1, 0.], [0., -2.], [0., -2.]]),
        ])
        r = evaluate_accuracy(euclidean_distance, anchor_embeddings, positive_embeddings, negative_embeddings)
        self.assertAllEqual(r, [True, False, True])


if __name__ == '__main__':
    tf.test.main()
