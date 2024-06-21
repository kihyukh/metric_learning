from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

from util.registry.model import Model


class InceptionModel(Model):
    name = 'inception'

    def __init__(self, conf, extra_info):
        super(InceptionModel, self).__init__(conf, extra_info)

        width = conf['image']['width'] if 'random_crop' not in conf['image'] else conf['image']['random_crop']['width']
        height = conf['image']['height'] if 'random_crop' not in conf['image'] else conf['image']['random_crop']['height']
        channel = conf['image']['channel']
        self.model = InceptionResNetV2(include_top=False,
                                       pooling='avg',
                                       weights='imagenet',
                                       input_shape=(width, height, channel))

    def preprocess_image(self, image):
        return preprocess_input(image)
