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

import bucket_api
import data_library
import data_library_query


IMAGES_FNAME = 'images.json'
LABELS_FNAME = 'labels.json'

def random_string(n):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(n))


class Dataset(object):
    """A Dataset that we can train on, using W&B Artifacts for tracking."""

    @classmethod
    def from_artifact(cls, artifact):
        """Given an artifact, construct a Dataset object."""
        artifact_dir = artifact.download()
        examples = json.load(open(os.path.join(artifact_dir, IMAGES_FNAME)))
        labels = json.load(open(os.path.join(artifact_dir, LABELS_FNAME)))
        return cls(examples, labels, artifact=artifact)

    @classmethod
    def from_library_query(cls, example_image_paths, label_types):
        """Query our data library to construct a Dataset object.

        You might make many different methods like this, to do different kinds of
        queries, or build datasets in different ways.
        """
        bucketapi = bucket_api.get_bucket_api()
        examples = [
            [path, bucketapi.get_hash(path)] for path in example_image_paths]
        labels = data_library_query.labels_for_images(
            example_image_paths, label_types)
        return cls(examples, labels)
    
    def __init__(self, examples, labels, artifact=None):
        """Constructor.

        A dataset consists of a list of examples, and their labels. We keep the
        examples and labels in sorted order, so that the same labels and examples
        will produce the same exact artifact.
        """
        self._artifact = artifact
        self.examples = sorted(examples)
        self.labels = sorted(labels, key=lambda l: (l['id'], 'bbox' in l))

    @property
    def example_image_paths(self):
        """Return the relative paths to example images for this dataset."""
        return set([e[0] for e in self.examples])

    def download(self):
        """Download the actual dataset contents."""
        # We use the external_data_dir directory of our artifact. This directory
        # is a good place to cache files that are related to the artifact, but not
        # actually part of the artifact's contents.
        datadir = self.artifact().external_data_dir
        bucketapi = bucket_api.get_bucket_api()
        for example in self.examples:
            image_path, image_hash = example
            bucketapi.download_file(image_path, os.path.join(datadir, image_path), hash=image_hash)
        return datadir

    def artifact(self):
        """Return an artifact that represents this dataset."""
        # Either just return the artifact we were constructed with, or build an
        # artifact and keep a reference to it.
        if self._artifact is None:
            self._artifact = self._to_artifact()
        return self._artifact

    def _to_artifact(self):
        """Build an artifact that represents this datset."""

        # You can put whatever you want in the artifacts metadata, this will be displayed
        # in tables in the W&B UI, and will be indexed for querying.
        annotation_types = []
        if 'bbox' in self.labels[0]:
            annotation_types.append('bbox')
        if 'segmentation' in self.labels[0]:
            annotation_types.append('segmentation')
        cat_counts = defaultdict(int)
        categories = data_library.get_categories()
        cats_by_id = {c['id']: c['name'] for c in categories}
        for l in self.labels:
            cat_name = cats_by_id[l['category_id']]
            cat_counts[cat_name] += 1

        # We use a WriteableArtifact, which gives us a directory to write our files into.
        artifact = wandb.WriteableArtifact(
            type='dataset',
            metadata= {
                'n_examples': len(self.examples),
                'annotation_types': annotation_types,
                'category_counts': cat_counts})

        with open(os.path.join(artifact.artifact_dir, IMAGES_FNAME), 'w') as f:
            json.dump(self.examples, f, indent=2, sort_keys=True)
        with open(os.path.join(artifact.artifact_dir, LABELS_FNAME), 'w') as f:
            json.dump(self.labels, f, indent=2, sort_keys=True)

        return artifact