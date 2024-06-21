import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.batch_design import BatchDesign
from util.tensor_operations import upper_triangular_part
from metric_learning.batch_designs.grouped import GroupedBatchDesign
from metric_learning.batch_designs.grouped import get_npair_distances

tf.enable_eager_execution()


class GroupedBatchDesignTest(tf.test.TestCase):
    def testGroupedDataset(self):
        image_files = ['a', 'b', 'c', 'd', 'e', 'f']
        labels = [3, 1, 2, 3, 1, 1]

        conf = {
            'batch_design': {
                'name': 'grouped',
                'group_size': 2,
                'batch_size': 4
            }
        }
        dataset: GroupedBatchDesign = BatchDesign.create('grouped', conf, {'data_loader': None})
        batch = dataset.get_next_batch(image_files, labels, conf['batch_design'])
        self.assertAllEqual(batch, [
            ('b', 1),
            ('f', 1),
            ('d', 3),
            ('a', 3),
        ])

    def testGetNpairDistances(self):
        embeddings = tf.constant([
            [1.],
            [2.],
            [5.],
            [7.],
        ])
        distances, matches = get_npair_distances(
            embeddings, 2, DistanceFunction.EUCLIDEAN_DISTANCE)
        self.assertAllEqual(distances, [
            [1., 6.],
            [3., 2.],
        ])

    def testNpairPairwiseWeights(self):
        labels = tf.constant([0, 0, 1, 1, 2, 2, 3, 3])
        npair = 2
        label_counts = [4, 4, 2, 2]
        extra_info = {
            'num_images': 20,
            'num_labels': 4,
            'label_counts': label_counts,
        }
        weights = GroupedBatchDesign.get_npair_pairwise_weights(labels, npair, extra_info)

        num_labels = extra_info['num_labels']
        num_average_images_per_label = extra_info['num_images'] / extra_info['num_labels']

        expected_weights = [
            [
                (num_average_images_per_label * num_average_images_per_label) /
                (label_counts[0] * (label_counts[0] - 1)),
                ((npair - 1) * num_average_images_per_label * num_average_images_per_label) /
                ((num_labels - 1) * label_counts[0] * label_counts[1]),
            ],
            [
                ((npair - 1) * num_average_images_per_label * num_average_images_per_label) /
                ((num_labels - 1) * label_counts[1] * label_counts[0]),
                (num_average_images_per_label * num_average_images_per_label) / (label_counts[1] * (label_counts[1] - 1)),
            ],
            [
                (num_average_images_per_label * num_average_images_per_label) / (label_counts[2] * (label_counts[2] - 1)),
                ((npair - 1) * num_average_images_per_label * num_average_images_per_label) /
                ((num_labels - 1) * label_counts[2] * label_counts[3]),
            ],
            [
                ((npair - 1) * num_average_images_per_label * num_average_images_per_label) /
                ((num_labels - 1) * label_counts[3] * label_counts[2]),
                (num_average_images_per_label * num_average_images_per_label) / (label_counts[3] * (label_counts[3] - 1)),
            ],
        ]
        self.assertAllClose(weights, expected_weights)

    def testNpairWeights(self):
        labels = tf.constant([0, 0, 1, 1, 2, 2, 3, 3])
        npair = 2
        label_counts = [4, 4, 2, 2]
        extra_info = {
            'num_images': 20,
            'num_labels': 4,
            'label_counts': label_counts,
        }
        weights = GroupedBatchDesign.get_npair_weights(labels, npair, extra_info)

        num_average_images_per_label = extra_info['num_images'] / extra_info['num_labels']
        expected_weights = [
            (num_average_images_per_label ** 3) / (label_counts[0] * (label_counts[0] - 1) * label_counts[1]),
            (num_average_images_per_label ** 3) / (label_counts[1] * (label_counts[1] - 1) * label_counts[0]),
            (num_average_images_per_label ** 3) / (label_counts[2] * (label_counts[2] - 1) * label_counts[3]),
            (num_average_images_per_label ** 3) / (label_counts[3] * (label_counts[3] - 1) * label_counts[2]),
        ]
        self.assertAllClose(weights, expected_weights)

    def testPairwiseWeights(self):
        labels = tf.constant([0, 0, 1, 1])
        group_size = 2
        num_groups = int(labels.shape[0]) // 2
        batch_size = num_groups * group_size
        label_counts = [4, 2]
        extra_info = {
            'num_images': 10,
            'num_labels': 2,
            'label_counts': label_counts,
        }
        weights = GroupedBatchDesign.get_pairwise_weights(labels, group_size, extra_info)

        num_labels = extra_info['num_labels']
        num_images = extra_info['num_images']

        expected_weights = [
            [
                0,
                (group_size - 1) * num_images * (num_images - 1) /
                ((batch_size - 1) * num_labels * label_counts[0] * (label_counts[0] - 1)),
                group_size * (num_groups - 1) * num_images * (num_images - 1) /
                ((batch_size - 1) * num_labels * (num_labels - 1) * label_counts[0] * label_counts[1]),
                group_size * (num_groups - 1) * num_images * (num_images - 1) /
                ((batch_size - 1) * num_labels * (num_labels - 1) * label_counts[0] * label_counts[1]),
            ],
            [
                0,
                0,
                group_size * (num_groups - 1) * num_images * (num_images - 1) /
                ((batch_size - 1) * num_labels * (num_labels - 1) * label_counts[0] * label_counts[1]),
                group_size * (num_groups - 1) * num_images * (num_images - 1) /
                ((batch_size - 1) * num_labels * (num_labels - 1) * label_counts[0] * label_counts[1]),
            ],
            [
                0,
                0,
                0,
                (group_size - 1) * num_images * (num_images - 1) /
                ((batch_size - 1) * num_labels * label_counts[1] * (label_counts[1] - 1)),
            ],
            [
                0,
                0,
                0,
                0,
            ],
        ]
        self.assertAllClose(
            upper_triangular_part(weights),
            upper_triangular_part(tf.constant(expected_weights)))


if __name__ == '__main__':
    tf.test.main()
