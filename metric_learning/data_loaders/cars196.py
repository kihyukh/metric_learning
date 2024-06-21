from util.registry.data_loader import DataLoader
from util.dataset import extract_tgz
from util.dataset import download
from util.config import CONFIG

import tensorflow as tf
import os
import scipy.io
import shutil


class Cars196DataLoader(DataLoader):
    name = 'cars196'

    def prepare_files(self):
        data_directory = os.path.join(CONFIG['dataset']['data_dir'], self.name)
        if tf.gfile.Exists(data_directory):
            count = 0
            for root, dirnames, filenames in os.walk(data_directory):
                for filename in filenames:
                    if filename.lower().endswith('.png'):
                        count += 1
            if count == 16185:
                return
        filepath = download(
            'http://imagenet.stanford.edu/internal/car196/car_ims.tgz',
            os.path.join(CONFIG['dataset']['temp_dir'], self.name)
        )
        extract_tgz(filepath, os.path.join(CONFIG['dataset']['temp_dir'], self.name))

        filepath = download(
            'http://imagenet.stanford.edu/internal/car196/cars_annos.mat',
            os.path.join(CONFIG['dataset']['temp_dir'], self.name)
        )
        annotation_mat = scipy.io.loadmat(filepath)
        annotations = {}
        for filename, x1, y1, x2, y2, class_id, is_training in annotation_mat['annotations'][0]:
            filename = filename[0].split('/')[-1]
            x1 = x1[0, 0]
            y1 = y1[0, 0]
            x2 = x2[0, 0]
            y2 = y2[0, 0]
            class_id = class_id[0, 0]
            is_training = is_training[0, 0]
            annotations[filename] = (class_id, x1, y1, x2, y2, is_training)

        train_data_directory = os.path.join(data_directory, 'train')
        test_data_directory = os.path.join(data_directory, 'test')
        if not tf.gfile.Exists(train_data_directory):
            tf.gfile.MakeDirs(train_data_directory)
        if not tf.gfile.Exists(test_data_directory):
            tf.gfile.MakeDirs(test_data_directory)
        for root, dirnames, filenames in os.walk(os.path.join(CONFIG['dataset']['temp_dir'], self.name, 'car_ims')):
            for filename in filenames:
                class_id, x1, y1, x2, y2, is_training = annotations[filename]
                if class_id <= 98:
                    if not tf.gfile.Exists(os.path.join(train_data_directory, str(class_id))):
                        tf.gfile.MakeDirs(os.path.join(train_data_directory, str(class_id)))
                    shutil.copyfile(
                        os.path.join(root, filename),
                        os.path.join(train_data_directory, str(class_id), filename))
                else:
                    if not tf.gfile.Exists(os.path.join(test_data_directory, str(class_id))):
                        tf.gfile.MakeDirs(os.path.join(test_data_directory, str(class_id)))
                    shutil.copyfile(
                        os.path.join(root, filename),
                        os.path.join(test_data_directory, str(class_id), filename))

    def image_parse_function(self, filename):
        width = self.conf['image']['width']
        height = self.conf['image']['height']
        channel = self.conf['image']['channel']
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=channel)
        image_resized = tf.image.resize_image_with_pad(image_decoded, target_height=height, target_width=width)
        return image_resized
