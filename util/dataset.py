from tqdm import tqdm

import math
import random
import requests
import tarfile
import tensorflow as tf
import os
import zipfile

from collections import defaultdict
from shutil import copyfile
from util.config import CONFIG


def download(url, directory, filename=None):
    if filename is None:
        filename = url.split('/')[-1]
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0));
    block_size = 1024
    wrote = 0
    with open(filepath, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print('ERROR, something went wrong')
    return filepath


def extract_tgz(filepath, directory):
    tar = tarfile.open(filepath, 'r')
    for item in tar:
        tar.extract(item, directory)


def extract_zip(filepath, directory):
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(directory)


def load_images_from_directory(directory, splits=None, distort=None, multiple=1):
    label_map = {}
    labels = []
    image_files = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if splits:
                split = int(subdir.split('/')[-2])
                if split not in splits:
                    continue
            label = subdir.split('/')[-1]
            if label not in label_map:
                label_map[label] = len(label_map)
            image_files.append(os.path.join(subdir, file))
            labels.append(label_map[label])
    if not distort:
        return image_files * multiple, labels * multiple
    label_file_map = defaultdict(list)
    for label, file in zip(labels, image_files):
        label_file_map[label].append(file)
    new_label_file_map = defaultdict(list)
    for index, (label, file_list) in enumerate(label_file_map.items()):
        if index < len(label_file_map) // 2:
            num_samples = distort['base'] - distort['value']
        else:
            num_samples = distort['base'] + distort['value']
        new_label_file_map[label] = file_list[0:num_samples]
    ret = []
    for label, file_list in new_label_file_map.items():
        ret += [(file, label) for file in file_list]
    return zip(*ret)


def split_train_test(images, labels, split_test_ratio=0.2):
    data = list(zip(images, labels))
    random.shuffle(data)
    num_test = int(split_test_ratio * len(data))
    training_data = data[:len(data) - num_test]
    testing_data = data[-num_test:]
    return training_data, testing_data


def split_train_test_by_label(images, labels, split_test_ratio=0.2):
    data = list(zip(images, labels))
    random.shuffle(data)

    data_map = defaultdict(list)
    for image, label in zip(images, labels):
        data_map[label].append(image)
    candidate_labels = list(data_map.keys())
    random.shuffle(candidate_labels)
    num_test = int(split_test_ratio * len(candidate_labels))
    training_labels = candidate_labels[:len(candidate_labels) - num_test]
    testing_labels = candidate_labels[-num_test:]
    training_data = filter(lambda x: x[1] in training_labels, data)
    testing_data = filter(lambda x: x[1] in testing_labels, data)
    return list(training_data), list(testing_data)


def save_image_files(image_files, save_dir):
    for image_file in image_files:
        dest = os.path.join(save_dir, '/'.join(image_file.split('/')[-2:]))
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        copyfile(image_file, dest)


def group_npairs(images, labels, n):
    ret = []
    for batch in range(int(images.shape[0]) // (n * 2)):
        cur_labels = labels[batch * n * 2: (batch + 1) * n * 2]
        cur_images = images[batch * n * 2: (batch + 1) * n * 2]
        data_map = defaultdict(list)
        for index, label in enumerate(cur_labels):
            data_map[int(label)].append(index)

        first_images = []
        second_images = []
        for label in data_map.keys():
            first_images.append(cur_images[data_map[label][0]])
            second_images.append(cur_images[data_map[label][1]])
        ret.append((
            tf.stack(first_images),
            tf.stack(second_images),
        ))
    return ret


def get_training_files_labels(conf):
    cv_splits = CONFIG['dataset'].getint('cross_validation_splits')
    cv_split = CONFIG['dataset'].getint('cross_validation_split')
    train_dir = os.path.join(
        CONFIG['dataset']['experiment_dir'],
        conf['dataset']['name'],
        'train')
    image_files, labels = load_images_from_directory(
        train_dir,
        splits=set(range(cv_splits)) - {cv_split},
        distort=conf['dataset'].get('distort'),
        multiple=conf['dataset'].get('multiple', 1),
    )
    if conf['dataset'].get('num_labels'):
        data_map = defaultdict(list)
        for image_file, label in zip(image_files, labels):
            data_map[label].append(image_file)
        data_map = dict(filter(lambda x: len(x[1]) >= 5, data_map.items()))
        sampled_labels = random.sample(list(data_map.keys()), conf['dataset']['num_labels'])
        sampled_label_map = {}
        for sampled_label in sampled_labels:
            index = len(sampled_label_map)
            sampled_label_map[sampled_label] = index
        ret_image_files = []
        ret_labels = []
        for original_index, new_index in sampled_label_map.items():
            ret_image_files += data_map[original_index]
            ret_labels += [new_index] * len(data_map[original_index])
        return ret_image_files, ret_labels
    else:
        return image_files, labels
