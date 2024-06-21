from util.dataset import download, extract_tgz, extract_zip
from util.registry.data_loader import DataLoader
from util.config import CONFIG

import tensorflow as tf
import shutil
import os


class LFWDataLoader(DataLoader):
    name = 'lfw'

    def prepare_files(self):
        data_directory = os.path.join(CONFIG['dataset']['data_dir'], self.name)
        if tf.gfile.Exists(data_directory):
            count = 0
            for root, dirnames, filenames in os.walk(data_directory):
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        count += 1
            if count == 507647:
                return
        filepath = download(
            'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
            os.path.join(CONFIG['dataset']['temp_dir'], self.name)
        )
        extract_path = os.path.join(CONFIG['dataset']['data_dir'], self.name, 'test')
        extract_tgz(filepath, extract_path)
        for root, dirnames, filenames in os.walk(os.path.join(extract_path, 'lfw')):
            for dirname in dirnames:
                shutil.move(os.path.join(root, dirname), os.path.join(extract_path, dirname))
        os.rmdir(os.path.join(extract_path, 'lfw'))

        pairs_filepath = download(
            'http://vis-www.cs.umass.edu/lfw/pairs.txt',
            os.path.join(CONFIG['dataset']['temp_dir'], 'pairs.txt'))
        shutil.move(pairs_filepath,
                    os.path.join(CONFIG['dataset']['data_dir'], self.name, 'pairs.txt'))

        webface_filepath = download(
            'https://s3-us-west-2.amazonaws.com/hominot/research/dataset/CASIA-WebFace.zip',
            os.path.join(CONFIG['dataset']['temp_dir'], 'webface')
        )
        extract_path = os.path.join(CONFIG['dataset']['data_dir'], self.name, 'train')
        extract_zip(webface_filepath, extract_path)
        for root, dirnames, filenames in os.walk(os.path.join(extract_path, 'CASIA-WebFace')):
            for dirname in dirnames:
                shutil.move(os.path.join(root, dirname), os.path.join(extract_path, dirname))
        os.rmdir(os.path.join(extract_path, 'CASIA-WebFace'))

    def image_parse_function(self, filename):
        width = self.conf['image']['width']
        height = self.conf['image']['height']
        channel = self.conf['image']['channel']
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=channel)
        image_resized = tf.image.resize_images(image_decoded, [width, height])
        return image_resized
