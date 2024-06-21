from util.dataset import download, extract_zip
from util.registry.data_loader import DataLoader
from shutil import copyfile
from util.config import CONFIG

import tensorflow as tf
import os


class StanfordOnlineProductDataLoader(DataLoader):
    name = 'stanford_online_product'

    def prepare_files(self):
        data_directory = os.path.join(CONFIG['dataset']['data_dir'], self.name)
        if tf.gfile.Exists(data_directory):
            count = 0
            for root, dirnames, filenames in os.walk(data_directory):
                for filename in filenames:
                    if filename.lower().endswith('.jpg'):
                        count += 1
            if count == 120053:
                return
        filepath = download(
            'https://s3-us-west-2.amazonaws.com/hominot/research/dataset/Stanford_Online_Products.zip',
            os.path.join(CONFIG['dataset']['temp_dir'], self.name)
        )
        extract_zip(filepath, os.path.join(CONFIG['dataset']['temp_dir'], self.name))
        extracted_path = os.path.join(CONFIG['dataset']['temp_dir'], self.name, 'Stanford_Online_Products')
        training_files = set()
        with open(os.path.join(extracted_path, 'Ebay_train.txt')) as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    continue
                _, _, _, filename = line.rstrip().split(' ')
                training_files.add(filename.split('/')[1])

        for directory in next(os.walk(extracted_path))[1]:
            for filename in os.listdir(os.path.join(extracted_path, directory)):
                if filename in training_files:
                    dest_directory = os.path.join(data_directory, 'train')
                else:
                    dest_directory = os.path.join(data_directory, 'test')
                object_id, index = filename.split('.')[0].split('_')
                if not tf.gfile.Exists(os.path.join(dest_directory, object_id)):
                    tf.gfile.MakeDirs(os.path.join(dest_directory, object_id))
                copyfile(os.path.join(extracted_path, directory, filename), os.path.join(dest_directory, object_id, filename))

    def image_parse_function(self, filename):
        width = self.conf['image']['width']
        height = self.conf['image']['height']
        channel = self.conf['image']['channel']
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=channel)
        image_resized = tf.image.resize_image_with_pad(image_decoded, target_height=height, target_width=width)
        return image_resized


if __name__ == '__main__':
    data_loader = DataLoader.create('stanford_online_product')
    data_loader.prepare_files()
