"""Code for versioning datasets with W&B Artifacts.

You should copy this file into your code base and customize it to your needs.

For each dataset version we create, we store an artifact in W&B. Each dataset
artifact contains a list of examples and their labels. We don't store the
actual image data in the artifact. Instead we keep the image data in our
data library (stored in a bucket in cloud storage).
"""
from collections import defaultdict
import json
import os
import random
import string

import wandb

import data_library
import data_library_query

IMAGES_FNAME = 'images.json'
LABELS_FNAME = 'labels.json'

def random_string(n):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(n))

def create_dataset(example_image_paths, label_types):
    """Create a dataset artifact by querying our data library."""

    # query our data library to get all labels for the examples we
    # want to include in this dataset
    labels = data_library_query.labels_for_images(
        example_image_paths, label_types)

    # compute counts of labels for each category, to store in our
    # dataset's metadata
    cat_counts = defaultdict(int)
    categories = data_library.get_categories()
    cats_by_id = {c['id']: c['name'] for c in categories}
    for l in labels:
        cat_name = cats_by_id[l['category_id']]
        cat_counts[cat_name] += 1

    # get the final set of example paths
    example_paths = set(l['image_path'] for l in labels)

    # Create our dataset artifact. We can store whatever we want in
    # metadata, for later querying.
    artifact = wandb.Artifact(
        type='dataset',
        metadata= {
            'n_examples': len(example_paths),
            'annotation_types': label_types,
            'category_counts': cat_counts})

    # Add all of the images to the dataset. We add them as references,
    # which means the artifact won't contain the images themselves.
    for example_path in example_paths:
        artifact.add_reference(
            data_library.get_absolute_path(example_path),
            name=os.path.join('images', example_path))

    # Store the labels file itself inside the artifact
    with artifact.new_file(LABELS_FNAME) as f:
        # We use sort_keys=True to get a consistent result
        json.dump(labels, f, indent=2, sort_keys=True)

    return artifact