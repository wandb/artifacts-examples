"""Query our data library, to produce a new dataset artifact."""

import argparse
import random
import sys

import wandb

import dataset
import data_library_query

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_name', type=str, required=True,
                    help='name for this dataset')

parser.add_argument('--supercategories', type=str, nargs='*', default=[],
                    help='coco supercategories to take examples from')
parser.add_argument('--categories', type=str, nargs='*', default=[],
                    help='coco categories to take examples from')
parser.add_argument('--select_fraction', type=float, default=1,
                    help='random fraction of examples to select')

parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--annotation_types', type=str, nargs='*', required=True,
                    choices=['bbox', 'segmentation'],
                    help='coco annotation types to include in dataset')


def main(argv):
    args = parser.parse_args()

    # Track dataset creation as a W&B run, so that we have a log of
    # how this dataset was created.
    run = wandb.init(job_type='create_dataset')
    run.config.update(args)

    random.seed(args.seed)

    # Query our data library for examples that have labels in the specified
    # categories or supercategories.
    chosen_cats = data_library_query.categories_filtered(
        args.supercategories, args.categories)
    chosen_cat_ids = [c['id'] for c in chosen_cats]
    labels = data_library_query.labels_of_types_and_categories(
        args.annotation_types, chosen_cat_ids)
    example_image_paths = set(l['image_path'] for l in labels)
    example_image_paths = set(random.sample(
        example_image_paths, int(args.select_fraction * len(example_image_paths))))
    if len(example_image_paths) == 0:
        print('Error, you must select at least 1 image')
        sys.exit(1)

    artifact = run.new_artifact(type='dataset', name=args.dataset_name, aliases=['latest'])

    # query our library to make the dataset, loading the examples and labels into our new
    # artifact
    dataset.Dataset.from_library_query(artifact, example_image_paths, args.annotation_types)

    # log the artifact to W&B
    artifact.save()


if __name__ == '__main__':
    main(sys.argv)