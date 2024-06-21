import tensorflow as tf

from util.registry.data_loader import DataLoader
from util.config import CONFIG

import argparse
import random
import shutil
import os


if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(
        description='Download dataset for training/testing models')
    parser.add_argument('--dataset', help='Name of the dataset')

    args = parser.parse_args()

    directory = os.path.join(CONFIG['dataset']['experiment_dir'], args.dataset)

    data_loader: DataLoader = DataLoader.create(args.dataset)
    data_loader.prepare_files()

    # copy test data
    shutil.rmtree(os.path.join(directory, 'test'), ignore_errors=True)
    shutil.copytree(
        os.path.join(CONFIG['dataset']['data_dir'], args.dataset, 'test'),
        os.path.join(directory, 'test'))

    # cross validation splits
    shutil.rmtree(os.path.join(directory, 'train'), ignore_errors=True)
    splits = list(range(CONFIG['dataset'].getint('cross_validation_splits')))
    for k in splits:
        if not tf.gfile.Exists(os.path.join(directory, 'train', str(k))):
            tf.gfile.MakeDirs(os.path.join(directory, 'train', str(k)))
    for root, dirnames, filenames in os.walk(os.path.join(CONFIG['dataset']['data_dir'], args.dataset, 'train')):
        dirnames = list(dirnames)
        random.shuffle(dirnames)
        for split in splits:
            start = int(split * len(dirnames) / len(splits))
            end = int((split + 1) * len(dirnames) / len(splits))
            for dirname in dirnames[start:end]:
                dest_dir = os.path.join(directory, 'train', str(split), dirname)
                shutil.copytree(os.path.join(root, dirname), dest_dir)
