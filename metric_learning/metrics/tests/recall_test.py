import tensorflow as tf

from metric_learning.metrics.recall import compute_recall
from metric_learning.metrics.recall import count_singletons
from metric_learning.constants.distance_function import get_distance_function


tf.enable_eager_execution()


class RecallMetricTest(tf.test.TestCase):
    def testSingletonCount(self):
        labels = [1, 1, 2, 2, 2, 3, 4, 4, 5]
        ret = count_singletons(labels)
        self.assertEqual(ret, 2)

    def testRecall(self):
        embeddings = tf.constant([
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
        ])
        labels = tf.constant([1, 1, 2, 2], tf.int64)
        for parametrization in ['euclidean_distance', 'cosine_similarity']:
            r = compute_recall([embeddings], [labels], [1], get_distance_function(parametrization))
            self.assertEqual(r, {1: 1.0})

    def testRecall2(self):
        embeddings = tf.constant([
            [1., 0.],
            [4., 0.],
            [0., 1.],
            [1., 2.],
        ])
        labels = tf.constant([1, 1, 2, 2], tf.int64)
        ret = compute_recall(
            [embeddings], [labels],
            [1, 2, 3],
            get_distance_function('euclidean_distance'))
        self.assertEqual(ret, {1: 0.5, 2: 0.75, 3: 1.0})
        ret = compute_recall(
            [embeddings], [labels],
            [1, 2, 3],
            get_distance_function('cosine_similarity'))
        self.assertEqual(ret, {1: 1.0, 2: 1.0, 3: 1.0})

    def testRecallMultipleBlocks(self):
        embeddings1 = tf.constant([
            [1., 0.],
            [4., 0.],
        ])
        embeddings2 = tf.constant([
            [0., 1.],
            [1., 2.],
        ])
        labels1 = tf.constant([1, 1], tf.int64)
        labels2 = tf.constant([2, 2], tf.int64)
        ret = compute_recall(
            [embeddings1, embeddings2], [labels1, labels2],
            [1, 2, 3],
            get_distance_function('euclidean_distance'))
        self.assertEqual(ret, {1: 0.5, 2: 0.75, 3: 1.0})
        ret = compute_recall(
            [embeddings1, embeddings2], [labels1, labels2],
            [1, 2, 3],
            get_distance_function('cosine_similarity'))
        self.assertEqual(ret, {1: 1.0, 2: 1.0, 3: 1.0})

    def testRecallWithSingleton(self):
        embeddings = tf.constant([
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
        ])
        labels = tf.constant([1, 1, 2, 3, 4], tf.int64)
        r = compute_recall(
            [embeddings], [labels],
            [1],
            get_distance_function('euclidean_distance'))
        self.assertEqual(r, {1: 1.0})


if __name__ == '__main__':
    tf.test.main()
