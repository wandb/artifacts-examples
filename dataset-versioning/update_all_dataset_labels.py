"""Create new versions of datasets that need updates."""

import argparse
import sys

import wandb

import dataset

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
        ds_artifact = run.use_artifact(type='dataset', name='%s:latest' % d.name)
        ds = dataset.Dataset.from_artifact(ds_artifact)

        # construct dataset artifact contents using the example in the loaded dataset,
        # but with the most recent labels from the library.
        artifact = wandb.Artifact(type='dataset', name=d.name, aliases=['latest'])
        dataset.Dataset.from_library_query(artifact, ds.example_image_paths,
                                           ds_artifact.metadata['annotation_types'])
        run.log_artifact(artifact)


if __name__ == '__main__':
    main(sys.argv)