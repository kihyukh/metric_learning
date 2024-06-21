import numpy as np
import tensorflow as tf

import datetime
import json
import math
import os

from decimal import Decimal
from tqdm import tqdm
from util.dataset import load_images_from_directory
from util.dataset import get_training_files_labels
from util.registry.data_loader import DataLoader
from util.registry.batch_design import BatchDesign
from util.registry.model import Model
from util.registry.metric import Metric
from util.logging import set_tensorboard_writer
from util.logging import upload_tensorboard_log_to_s3
from util.logging import create_checkpoint
from util.logging import save_config
from util.logging import db
from util.config import CONFIG


def evaluate(conf, model, data_files, train_stat):
    data = {}
    data_loader = DataLoader.create(conf['dataset']['name'], conf)
    Metric.cache.clear()
    with tf.contrib.summary.always_record_summaries():
        for metric_conf in model.conf['metrics']:
            if metric_conf['name'] == 'nmi' and conf['dataset']['name'] == 'stanford_online_product':
                continue
            conf_copy = {}
            conf_copy.update(conf)
            conf_copy['batch_design'] = metric_conf['batch_design']
            batch_design = BatchDesign.create(
                metric_conf['batch_design']['name'],
                conf_copy,
                {'data_loader': data_loader})
            metric = Metric.create(metric_conf['name'], conf)
            image_files, labels = data_files[metric_conf.get('dataset', 'test')]
            test_dataset, num_testcases = batch_design.create_dataset(
                model, image_files, labels,
                metric_conf['batch_design'], testing=True)
            score = metric.compute_metric(model, test_dataset, num_testcases)
            if type(score) is dict:
                for metric, s in score.items():
                    tf.contrib.summary.scalar('test ' + metric, s)
                    print('{}: {}'.format(metric, s))
                    data[metric] = Decimal(str(s))
            else:
                tf.contrib.summary.scalar('{}'.format(metric_conf['name']), score)
                print('{}: {}'.format(metric_conf['name'], score))
                data[metric_conf['name']] = Decimal(str(score))
    Metric.cache.clear()
    if CONFIG['tensorboard'].getboolean('dynamodb_upload'):
        dt = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
        table = db.Table('TrainHistory')
        item = {'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S')}
        item.update(train_stat)
        item.update(data)
        table.put_item(Item=item)
    return data


def stopping_criteria(metrics):
    if len(metrics) <= 3:
        return False
    if 'vrf@1' in metrics[0]:
        vrf_1 = [float(x['vrf@1']) for x in metrics]
        max_vrf_1 = max(vrf_1)
        if vrf_1[-1] > max_vrf_1:
            return False
    recall_1 = [float(x['recall@1']) for x in metrics]
    max_recall_1 = max(recall_1)
    if recall_1[-1] < max_recall_1 * 0.93:
        return True
    if recall_1[-1] < recall_1[-2] * 0.97 and recall_1[-2] < recall_1[-3] * 0.97:
        return True
    if all([x * 1.003 < max_recall_1 for x in recall_1[-4:]]):
        return True
    return False


def get_metric_to_report(metrics):
    ret = {}
    recall_1 = [float(x['recall@1']) for x in metrics]
    for k, v in metrics[np.argmax(recall_1)].items():
        if k.startswith('recall@'):
            ret[k] = v
    if 'vrf@1' in metrics[0]:
        vrf_1 = [float(x['vrf@1']) for x in metrics]
        for k, v in metrics[np.argmax(vrf_1)].items():
            if k.startswith('vrf@'):
                ret[k] = v
    if 'auc' in metrics[0]:
        ret['auc'] = max([x['auc'] for x in metrics])
    if 'nmi' in metrics[0]:
        ret['nmi'] = max([x['nmi'] for x in metrics])
    return ret


def train(conf, experiment_name):
    print(json.dumps(conf, indent=4))
    data_loader = DataLoader.create(conf['dataset']['name'], conf)
    if conf['dataset']['cross_validation_split'] != -1:
        test_dir = os.path.join(CONFIG['dataset']['experiment_dir'],
                                      conf['dataset']['name'],
                                      'train',
                                      str(conf['dataset']['cross_validation_split']))
    else:
        test_dir = os.path.join(CONFIG['dataset']['experiment_dir'],
                                conf['dataset']['name'],
                                'test')
    data_files = {
        'test': load_images_from_directory(test_dir),
        'train': get_training_files_labels(conf),
    }

    writer, run_name = set_tensorboard_writer(conf, experiment_name)
    if not experiment_name:
        experiment_name = run_name.rsplit('_', 1)[0]
    writer.set_as_default()
    save_config(conf, run_name, experiment_name)

    dataset = BatchDesign.create(
        conf['batch_design']['name'], conf, {'data_loader': data_loader})

    train_images, train_labels = data_files['train']
    num_train_labels = max(train_labels) + 1
    label_counts = [0] * num_train_labels
    for label in train_labels:
        label_counts[label] += 1
    extra_info = {
        'num_labels': num_train_labels,
        'num_images': len(train_images),
        'label_counts': label_counts,
    }
    model = Model.create(conf['model']['name'], conf, extra_info)
    optimizers = {
        k: tf.train.AdamOptimizer(learning_rate=v) for
        k, (v, _) in model.learning_rates().items()
    }
    checkpoint = tf.train.Checkpoint(model=model)

    batch_design_conf = conf['batch_design']
    train_stat = {
        'id': run_name,
        'epoch': 0,
    }
    if CONFIG['train'].getboolean('initial_evaluation'):
        evaluate(conf, model, data_files, train_stat)
    step_counter = tf.train.get_or_create_global_step()
    step_counter.assign(0)

    metrics = []
    for epoch in range(conf['trainer']['num_epochs']):
        if conf['dataset'].get('num_labels'):
            train_images, train_labels = get_training_files_labels(conf)
        train_ds, num_examples = dataset.create_dataset(
            model,
            train_images,
            train_labels,
            batch_design_conf)
        train_ds = train_ds.batch(batch_design_conf['batch_size'],
                                  drop_remainder=True)
        batches = tqdm(train_ds,
                       total=math.ceil(num_examples / batch_design_conf['batch_size']),
                       desc='epoch #{}'.format(epoch + 1),
                       dynamic_ncols=True)
        losses = []
        batches_combined = 0
        grads = None
        for batch in batches:
            with tf.contrib.summary.record_summaries_every_n_global_steps(
                    CONFIG['tensorboard'].getint('record_every_n_global_steps'),
                    global_step=step_counter):
                with tf.GradientTape() as tape:
                    loss_value = model.loss(batch, model, dataset)
                    losses.append(float(loss_value))
                    batches.set_postfix({'loss': float(loss_value)})
                    tf.contrib.summary.scalar('loss', loss_value)

                if grads is None:
                    grads = tape.gradient(loss_value, model.variables)
                else:
                    for idx, g in enumerate(tape.gradient(loss_value, model.variables)):
                        if g is None:
                            continue
                        if grads[idx] is None:
                            grads[idx] = g
                        else:
                            grads[idx] += g
                batches_combined += 1
                if batches_combined == conf['batch_design']['combine_batches']:
                    for optimizer_key, (_, variables) in model.learning_rates().items():
                        filtered_grads = filter(lambda x: x[1] in variables, zip(grads, model.variables))
                        optimizers[optimizer_key].apply_gradients(filtered_grads)
                    batches_combined = 0
                    grads = None
                    step_counter.assign_add(1)
                if CONFIG['tensorboard'].getboolean('s3_upload') and \
                        int(step_counter) % int(CONFIG['tensorboard']['s3_upload_period']) == 0:
                    upload_tensorboard_log_to_s3(run_name)
        print('epoch #{} checkpoint: {}'.format(epoch + 1, run_name))
        if CONFIG['tensorboard'].getboolean('enable_checkpoint'):
            create_checkpoint(checkpoint, run_name, CONFIG['tensorboard'].getboolean('s3_upload'))
        train_stat['epoch'] = epoch + 1
        train_stat['loss'] = Decimal(str(sum(losses) / len(losses)))
        print('average loss: {:.4f}'.format(sum(losses) / len(losses)))
        if not conf['trainer']['evaluate_once']:
            metrics.append(evaluate(conf, model, data_files, train_stat))
        if conf['trainer']['early_stopping'] and stopping_criteria(metrics):
            break
    if conf['dataset']['name'] == 'stanford_online_product':
        create_checkpoint(checkpoint, run_name, s3_upload=True)
    if conf['trainer']['evaluate_once']:
        final_metrics = evaluate(conf, model, data_files, train_stat)
    else:
        final_metrics = get_metric_to_report(metrics)
    if CONFIG['tensorboard'].getboolean('dynamodb_upload'):
        table = db.Table('Experiment')
        for metric, score in final_metrics.items():
            table.update_item(
                Key={
                    'id': run_name,
                },
                UpdateExpression='SET #m = :u',
                ExpressionAttributeNames={
                    '#m': 'metric:{}'.format(metric),
                },
                ExpressionAttributeValues={
                    ':u': score,
                },
            )
