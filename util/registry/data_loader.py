from util.registry.class_registry import ClassRegistry

import tensorflow as tf


class DataLoader(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.data_loaders'

    def __init__(self, conf, extra_info):
        super(DataLoader, self).__init__()
        self.conf = conf
        self.extra_info = extra_info

    def prepare_files(self):
        raise NotImplementedError

    def image_parse_function(self, filename):
        raise NotImplementedError

    def random_crop(self, image):
        width = self.conf['image']['random_crop']['width']
        height = self.conf['image']['random_crop']['height']
        channel = self.conf['image']['channel']
        return tf.random_crop(image, [height, width, channel])

    def random_flip(self, image):
        return tf.image.random_flip_left_right(image)

    def center_crop(self, image):
        crop_width = self.conf['image']['random_crop']['width']
        crop_height = self.conf['image']['random_crop']['height']
        width = self.conf['image']['width']
        height = self.conf['image']['height']
        return tf.image.crop_to_bounding_box(
            image,
            (height - crop_height) // 2,
            (width - crop_width) // 2,
            crop_height,
            crop_width)

    def __str__(self):
        return self.conf['dataset']['name']
