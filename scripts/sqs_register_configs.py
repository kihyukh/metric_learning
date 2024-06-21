import argparse
import boto3
import json

from metric_learning.example_configurations import configs
from util.config import generate_configs_from_experiment

conn_sqs = boto3.resource('sqs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train using a specified config')
    parser.add_argument('--config', help='config to run')
    parser.add_argument('--experiment', help='experiment to run')
    args = parser.parse_args()

    queue = conn_sqs.get_queue_by_name(QueueName='experiment-configs')

    if args.experiment:
        experiments = generate_configs_from_experiment(args.experiment)
        experiment_name = args.experiment
    else:
        experiments = [configs[args.config]]
        experiment_name = ''

    for experiment in experiments:

        conf = json.dumps(experiment, indent=4)
        queue.send_message(
            MessageBody=conf,
            MessageAttributes={
                'experiment_name': {
                    'StringValue': experiment_name,
                    'DataType': 'String'
                }
            }
        )
