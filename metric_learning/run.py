import tensorflow as tf

import argparse
import copy

from metric_learning.example_configurations import configs
from util.config import generate_configs_from_experiment
from util.train import train

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Train using a specified config')
    parser.add_argument('--config', help='config to run')
    parser.add_argument('--experiment', help='experiment to run')
    parser.add_argument('--experiment_name', help='name of the experiment for logging')
    parser.add_argument('--experiment_index',
                        help='index of the experiment to run',
                        type=int)
    parser.add_argument('--split',
                        help='cross validation split number to use as validation data',
                        default=None,
                        type=int)
    args = parser.parse_args()

    if args.experiment:
        experiments = generate_configs_from_experiment(args.experiment)
        conf = copy.deepcopy(experiments[args.experiment_index])
        if args.split is not None:
            conf['dataset']['cross_validation_split'] = args.split
        train(conf, args.experiment_name)
    else:
        conf = copy.deepcopy(configs[args.config])
        if args.split is not None:
            conf['dataset']['cross_validation_split'] = args.split
        train(conf, args.experiment_name)
