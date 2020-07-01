"""Create new versions of datasets that need updates."""

import argparse
import collections
import json
import os
import random
import shutil
import sys
import tempfile

import wandb

import dataset
import data_library

parser = argparse.ArgumentParser(description='Create new versions of datasets if new labels are available.')


def main(argv):
    args = parser.parse_args()

    api = wandb.Api()

    # Initialize a W&B run
    run = wandb.init(job_type='update_dataset')
    run.config.update(args)

    # iterate through all the datasets we have.
    dataset_type = api.artifact_type('dataset')
    for d in dataset_type.artifact_collections():
        print('Checking latest for dataset: %s' % d)

        # fetch the latest version of each dataset artifact and download it's contents
        ds_artifact = run.use_artifact('%s:latest' % d.name)
        labels = json.load(open(ds_artifact.get_path('labels.json').download()))
        example_paths = set(l['image_path'] for l in labels)

        # construct dataset artifact contents using the example in the loaded dataset,
        # but with the most recent labels from the library.
        library_ds_artifact = dataset.create_dataset(
            d.name,
            example_paths,
            ds_artifact.metadata['annotation_types'])

        # If the digests aren't equal, then the labels have been updated, save the
        # library dataset artifact as the new versoin for this dataset.
        if ds_artifact.digest != library_ds_artifact.digest:
            print('  updated, create new dataset version')
            run.log_artifact(library_ds_artifact)


if __name__ == '__main__':
    main(sys.argv)