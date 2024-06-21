import tensorflow as tf

from metric_learning.batch_designs.pair import PairBatchDesign

tf.enable_eager_execution()


class PairBatchDesignTest(tf.test.TestCase):
    def testPairWeights(self):
        labels = tf.constant([0, 0, 1, 1, 2, 3, 4, 5])
        label_counts = [4, 4, 2, 2, 4, 4]
        extra_info = {
            'num_images': 30,
            'num_labels': 6,
            'label_counts': label_counts,
        }
        num_images = extra_info['num_images']
        num_labels = extra_info['num_labels']
        weights = PairBatchDesign.get_pairwise_weights(labels, 0.5, extra_info)

        evens = tf.range(labels.shape[0] // 2, dtype=tf.int64) * 2
        odds = tf.range(labels.shape[0] // 2, dtype=tf.int64) * 2 + 1
        even_labels = tf.gather(labels, evens)
        odd_labels = tf.gather(labels, odds)
        self.assertAllEqual(even_labels, [0, 1, 2, 4])
        self.assertAllEqual(odd_labels, [0, 1, 3, 5])
        positive_ratio = 0.5
        self.assertEqual(positive_ratio, 0.5)

        expected_weights = [
            (positive_ratio * num_images * (num_images - 1)) /
            (num_labels * label_counts[0] * (label_counts[0] - 1)),
            (positive_ratio * num_images * (num_images - 1)) /
            (num_labels * label_counts[1] * (label_counts[1] - 1)),
            ((1 - positive_ratio) * num_images * (num_images - 1)) /
            (num_labels * (num_labels - 1) * label_counts[2] * label_counts[3]),
            ((1 - positive_ratio) * num_images * (num_images - 1)) /
            (num_labels * (num_labels - 1) * label_counts[4] * label_counts[5]),
        ]

        self.assertAllClose(weights, expected_weights)


if __name__ == '__main__':
    tf.test.main()
