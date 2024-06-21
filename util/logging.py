import tensorflow as tf
import os

import boto3
import datetime
import json

from util.config import CONFIG


s3 = boto3.client('s3')
db = boto3.resource('dynamodb')


def get_run_name(conf):
    return '_'.join([
        conf['dataset']['name'],
        conf['model']['name'],
        conf['loss']['name'],
    ])


def set_tensorboard_writer(conf, experiment_name):
    if not experiment_name:
        run_name = get_run_name(conf)
    else:
        run_name = experiment_name

    local_tensorboard_dir = os.path.join(CONFIG['tensorboard']['local_dir'], 'tensorboard')
    if not tf.gfile.Exists(local_tensorboard_dir):
        tf.gfile.MakeDirs(local_tensorboard_dir)
    dt = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    run_dir = '{}_{}'.format(run_name, dt.strftime('%Y%m%d%H%M%S-%f'))

    if CONFIG['tensorboard'].getboolean('s3_upload'):
        s3.put_object(
            Bucket=CONFIG['tensorboard']['s3_bucket'],
            Body='',
            Key='{}/tensorboard/{}/'.format(CONFIG['tensorboard']['s3_key'], run_dir)
        )

    print('Starting {}'.format(run_dir))
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(local_tensorboard_dir, run_dir),
        flush_millis=10000)
    return writer, run_dir


def upload_tensorboard_log_to_s3(run_name):
    run_dir = os.path.join(CONFIG['tensorboard']['local_dir'], 'tensorboard', run_name)
    for filename in os.listdir(run_dir):
        s3.upload_file(
            os.path.join(run_dir, filename),
            CONFIG['tensorboard']['s3_bucket'],
            '{}/tensorboard/{}/{}'.format(CONFIG['tensorboard']['s3_key'], run_name, filename))


def save_config(conf, run_name, experiment_name):
    if CONFIG['tensorboard'].getboolean('s3_upload'):
        upload_string_to_s3(
            bucket=CONFIG['tensorboard']['s3_bucket'],
            body=json.dumps(conf, indent=4),
            key='{}/experiments/{}/config.json'.format(CONFIG['tensorboard']['s3_key'], run_name)
        )
    else:
        config_dir = os.path.join(CONFIG['tensorboard']['local_dir'], 'experiments', run_name)
        if not tf.gfile.Exists(config_dir):
            tf.gfile.MakeDirs(config_dir)
        with open(os.path.join(config_dir, 'config.json'), 'w') as f:
            json.dump(conf, f, indent=4)
    if CONFIG['tensorboard'].getboolean('dynamodb_upload'):
        dt = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
        data = {
            'id': run_name,
            'experiment_name': experiment_name,
            'config': json.dumps(conf),
            'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
        }
        for key, conf_dict in conf.items():
            if key in ['image', 'metrics']:
                continue
            for name, value in conf_dict.items():
                data['{}:{}'.format(key, name)] = str(value)

        table = db.Table('Experiment')
        table.put_item(Item=data)


def upload_file_to_s3(file_path, bucket, key):
    s3.upload_file(file_path, bucket, key)


def upload_string_to_s3(body, bucket, key):
    s3.put_object(Bucket=bucket, Body=body, Key=key)


def create_checkpoint(checkpoint, run_name, s3_upload):
    prefix = '{}/experiments/{}/checkpoints/ckpt'.format(
        CONFIG['tensorboard']['local_dir'],
        run_name)
    if not tf.gfile.Exists(os.path.dirname(prefix)):
        tf.gfile.MakeDirs(os.path.dirname(prefix))
    checkpoint.save(file_prefix=prefix)
    g = open(os.path.join(os.path.dirname(prefix), 'checkpoint_compat'), 'w')
    with open(os.path.join(os.path.dirname(prefix), 'checkpoint'), 'r') as f:
        for line in f:
            print(line.rstrip('\n').replace(os.path.dirname(prefix) + '/', ''), file=g)
    g.close()
    if s3_upload:
        for root, dirnames, filenames in os.walk(os.path.dirname(prefix)):
            for filename in filenames:
                if filename == 'checkpoint':
                    continue
                dest_filename = filename
                if filename == 'checkpoint_compat':
                    dest_filename = 'checkpoint'
                s3.upload_file(
                    os.path.join(root, filename),
                    CONFIG['tensorboard']['s3_bucket'],
                    '{}/experiments/{}/checkpoints/{}'.format(
                        CONFIG['tensorboard']['s3_key'],
                        run_name,
                        dest_filename
                    )
                )
                os.remove(os.path.join(root, filename))
