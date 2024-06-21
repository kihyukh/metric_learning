import tensorflow as tf

import argparse
import boto3
import json
import os
import tempfile

from decimal import Decimal
from util.config import CONFIG
from util.dataset import load_images_from_directory

from util.registry.model import Model
from util.registry.metric import Metric
from util.registry.data_loader import DataLoader
from util.registry.batch_design import BatchDesign


s3 = boto3.client('s3')
db = boto3.resource('dynamodb')
table = db.Table('Experiment')


def get_config(experiment):
    ret = table.get_item(Key={'id': experiment})
    return json.loads(ret['Item']['config'])


def get_checkpoint(temp_dir, experiment, step=None):
    if step is None:
        checkpoint_path = tf.train.latest_checkpoint(
            's3://hominot/research/metric_learning/experiments/{}/checkpoints'.format(experiment))
    else:
        checkpoint_path = 's3://hominot/research/metric_learning/experiments/{}/checkpoints/ckpt-{}'.format(
            experiment, step)
    for data in s3.list_objects(Bucket=CONFIG['tensorboard']['s3_bucket'],
                                Prefix=checkpoint_path.split('/', 3)[-1])['Contents']:
        s3.download_file(
            Bucket=CONFIG['tensorboard']['s3_bucket'],
            Key=data['Key'],
            Filename=os.path.join(temp_dir, data['Key'].split('/')[-1]))
    return os.path.join(temp_dir, checkpoint_path.split('/')[-1])


METRICS = [
    {
        'name': 'nmi',
        'batch_design': {
            'name': 'vanilla',
            'batch_size': 48,
        },
        'dataset': 'test',
    },
]

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Train using a specified config')
    parser.add_argument('--experiment', help='experiment id')
    parser.add_argument('--step', help='batch number')
    args = parser.parse_args()

    conf = get_config(args.experiment)
    conf['metrics'] = METRICS
    model = Model.create(conf['model']['name'], conf)

    test_dir = os.path.join(
        CONFIG['dataset']['experiment_dir'], conf['dataset']['name'], 'test')
    testing_files, testing_labels = load_images_from_directory(test_dir)

    data_loader = DataLoader.create(conf['dataset']['name'], conf)
    vanilla_ds = BatchDesign.create(
        'vanilla',
        conf,
        {'data_loader': data_loader})
    test_datasets = {
        'test': vanilla_ds.create_dataset(
            model, testing_files, testing_labels, conf['batch_design'], testing=True),
    }

    checkpoint = tf.train.Checkpoint(model=model)
    data = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        c = get_checkpoint(temp_dir, args.experiment, args.step)
        checkpoint.restore(c)

        for metric_conf in METRICS:
            metric = Metric.create(metric_conf['name'], conf)
            test_dataset, test_num_testcases = test_datasets[metric_conf['dataset']]
            score = metric.compute_metric(model, test_dataset, test_num_testcases)
            if type(score) is dict:
                for metric, s in score.items():
                    tf.contrib.summary.scalar('test ' + metric, s)
                    print('{}: {}'.format(metric, s))
                    data[metric] = Decimal(str(s))
            else:
                tf.contrib.summary.scalar('{}'.format(metric_conf['name']), score)
                print('{}: {}'.format(metric_conf['name'], score))
                data[metric_conf['name']] = Decimal(str(score))

    for metric, score in data.items():
        table.update_item(
            Key={
                'id': args.experiment,
            },
            UpdateExpression='SET #m = :u',
            ExpressionAttributeNames={
                '#m': 'metric:{}'.format(metric),
            },
            ExpressionAttributeValues={
                ':u': score,
            },
        )
