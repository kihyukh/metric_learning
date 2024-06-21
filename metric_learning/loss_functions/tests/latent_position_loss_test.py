import tensorflow as tf

from util.tensor_operations import pairwise_euclidean_distance_squared

tf.enable_eager_execution()


class LatentPositionLossTest(tf.test.TestCase):
    # TODO: add more relevant test cases
    def testPairwiseDifference(self):
        embeddings = tf.constant([
            [0, 1],
            [0, 2],
            [0, 3],
        ])
        y = pairwise_euclidean_distance_squared(embeddings, embeddings)
        self.assertAllEqual(y, [
            [0, 1, 4],
            [1, 0, 1],
            [4, 1, 0],
        ])


if __name__ == '__main__':
    tf.test.main()
