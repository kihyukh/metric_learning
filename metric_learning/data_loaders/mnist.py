from util.registry.data_loader import DataLoader

import gzip
import os
import shutil
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from collections import defaultdict
from util.config import CONFIG


class MNISTDataLoader(DataLoader):
    name = 'mnist'

    def prepare_files(self):
        def read32(bytestream):
            """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
            dt = np.dtype(np.uint32).newbyteorder('>')
            return np.frombuffer(bytestream.read(4), dtype=dt)[0]

        def check_image_file_header(filename):
            """Validate that filename corresponds to images for the MNIST dataset."""
            with tf.gfile.Open(filename, 'rb') as f:
                magic = read32(f)
                read32(f)  # num_images, unused
                rows = read32(f)
                cols = read32(f)
                if magic != 2051:
                    raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                                   f.name))
                if rows != 28 or cols != 28:
                    raise ValueError(
                        'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                        (f.name, rows, cols))

        def check_labels_file_header(filename):
            """Validate that filename corresponds to labels for the MNIST dataset."""
            with tf.gfile.Open(filename, 'rb') as f:
                magic = read32(f)

                if magic != 2049:
                    raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                                   f.name))

        def download(filename):
            """Download (and unzip) a file from the MNIST dataset if not already done."""
            filepath = os.path.join(CONFIG['dataset']['temp_dir'], self.name, filename)
            if tf.gfile.Exists(filepath):
                return filepath
            if not tf.gfile.Exists(os.path.join(CONFIG['dataset']['temp_dir'], self.name)):
                tf.gfile.MakeDirs(os.path.join(CONFIG['dataset']['temp_dir'], self.name))
            # CVDF mirror of http://yann.lecun.com/exdb/mnist/
            url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
            _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
            print('Downloading %s to %s' % (url, zipped_filepath))
            urllib.request.urlretrieve(url, zipped_filepath)
            with gzip.open(zipped_filepath, 'rb') as f_in, \
                    tf.gfile.Open(filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(zipped_filepath)
            return filepath

        def prepare_image_files(images_file, labels_file):
            """Download and parse MNIST dataset."""

            data_directory = os.path.join(CONFIG['dataset']['data_dir'], self.name)
            if tf.gfile.Exists(data_directory):
                count = 0
                for root, dirnames, filenames in os.walk(data_directory):
                    for filename in filenames:
                        if filename.endswith('.png'):
                            count += 1
                if count == 60000:
                    return

            images_file = download(images_file)
            labels_file = download(labels_file)

            check_image_file_header(images_file)
            check_labels_file_header(labels_file)

            with open(images_file, 'rb') as f:
                image_data = np.frombuffer(f.read(), np.uint8, offset=16)
            image_data = image_data.reshape(-1, 28, 28)

            with open(labels_file, 'rb') as f:
                label_data = np.frombuffer(f.read(), np.uint8, offset=8)

            label_count_map = defaultdict(int)
            for image, label in zip(image_data, label_data):
                label_count_map[label] += 1
                image = tf.reshape(tf.constant(image), [28, 28, 1])
                filename = '{:04d}.png'.format(label_count_map[label])
                if str(label) in ['5', '9']:
                    label_directory = os.path.join(data_directory, 'test', str(label))
                else:
                    label_directory = os.path.join(data_directory, 'train', str(label))
                if not tf.gfile.Exists(label_directory):
                    tf.gfile.MakeDirs(label_directory)
                tf.write_file(os.path.join(label_directory, filename), tf.image.encode_png(image))

        prepare_image_files('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')

    def image_parse_function(self, filename):
        width = self.conf['image']['width']
        height = self.conf['image']['height']
        channel = self.conf['image']['channel']
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=channel)
        image_resized = tf.image.resize_images(image_decoded, [width, height])
        return image_resized
