from util.registry.data_loader import DataLoader
from util.config import CONFIG

from mtcnn.mtcnn import MTCNN

import cv2
import tensorflow as tf
import shutil
import os


class LFWMtcnnDataLoader(DataLoader):
    name = 'lfw_mtcnn'

    def prepare_files(self):
        lfw_path = os.path.join(CONFIG['dataset']['data_dir'], 'lfw')
        detector = MTCNN()
        for root, dirnames, filenames in os.walk(lfw_path):
            for file in filenames:
                file_path = os.path.join(root, file)
                dest_path = file_path.replace('/lfw/', '/{}/'.format(self.name))
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                img = cv2.imread(file_path)
                result = None
                if img is not None:
                    height, width, channels = img.shape
                    result = detector.detect_faces(img)
                    image_string = tf.read_file(file_path)
                    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
                    result = list(filter(lambda x: x['box'][0] > 0 and x['box'][1] > 0, result))
                    result = list(filter(lambda x: x['box'][0] + x['box'][2] < width and x['box'][1] + x['box'][3] < height,
                                         result))
                if result:
                    result.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)
                    cropped = tf.image.crop_to_bounding_box(
                        image_decoded,
                        result[0]['box'][1],
                        result[0]['box'][0],
                        result[0]['box'][3],
                        result[0]['box'][2],
                    )
                    image_string = tf.image.encode_jpeg(cropped)
                    tf.write_file(dest_path, image_string)
                else:
                    shutil.copy(
                        file_path,
                        dest_path,
                    )

    def image_parse_function(self, filename):
        width = self.conf['image']['width']
        height = self.conf['image']['height']
        channel = self.conf['image']['channel']
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=channel)
        image_resized = tf.image.resize_image_with_pad(image_decoded, target_height=height, target_width=width)
        return image_resized
