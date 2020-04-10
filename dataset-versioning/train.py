"""Train a (fake) model on one of our datasets"""

import argparse
import collections
import os
import random
import sys
import tempfile

import wandb

import dataset
import bucket_api
import data_library

parser = argparse.ArgumentParser(description='Train a new model, based on a dataset artifact.')
parser.add_argument('--dataset', type=str, required=True, help='')
parser.add_argument('--model_type', required=True,
    type=str, choices=['bbox', 'segmentation'], help='')


def main(argv):
    args = parser.parse_args()

    run = wandb.init(job_type='train-%s' % args.model_type)
    run.config.update(args)

    # Get the artifact from W&B and mark it as input to this run.
    ds_artifact = run.use_artifact(type='dataset', name=args.dataset)
    if args.model_type not in ds_artifact.metadata['annotation_types']:
        print('Dataset %s has annotations %s, can\'t train model type: %s' % (
            args.dataset, ds_artifact.metadata['annotation_types'], args.model_type))
        sys.exit(1)

    # Initialize our Dataset class from the artifact's contents, and download
    # the actual dataset examples.
    ds = dataset.Dataset.from_artifact(ds_artifact)
    dataset_dir = ds.download()

    # Build X (images) and y (labels) for training
    X, y = [], []
    for label in ds.labels:
        X.append(os.path.join(dataset_dir, label['image_path']))
        y.append(label['bbox'])
    print ('Xlen, ylen', len(X), len(y))

    # Simulate training by logging a fake loss curve
    for i in range(10):
        run.log({'loss': random.random() / (i + 1)})

    # Save our trained model as an artifact. Here we save the artifact by passing
    # a path to the file we want to include in the artifact's contents.
    model_file = open('model.json', 'w')
    model_file.write(
        'This is a placeholder. In a real job, you\'d save model weights here\n%s\n' % random.random())
    model_file.close()

    run.log_artifact(
        type='model',
        name=args.model_type,
        paths='model.json',
        aliases='latest')

if __name__ == '__main__':
    main(sys.argv)