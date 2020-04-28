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
        return cls(examples, labels, artifact)

    @classmethod
    def from_library_query(cls, artifact, example_image_paths, label_types):
        """Query our data library to construct a Dataset object.

        You might make many different methods like this, to do different kinds of
        queries, or build datasets in different ways.
        """
        bucketapi = bucket_api.get_bucket_api()
        examples = sorted([
            [path, bucketapi.get_hash(path)] for path in example_image_paths])
        labels = sorted(data_library_query.labels_for_images(
            example_image_paths, label_types), key=lambda l: (l['id'], 'bbox' in l))

        cls._load_artifact(artifact, examples, labels)
        return cls(examples, labels, artifact)

    @classmethod
    def _load_artifact(cls, artifact, examples, labels):
        # You can put whatever you want in the artifacts metadata, this will be displayed
        # in tables in the W&B UI, and will be indexed for querying.
        annotation_types = []
        if 'bbox' in labels[0]:
            annotation_types.append('bbox')
        if 'segmentation' in labels[0]:
            annotation_types.append('segmentation')
        cat_counts = defaultdict(int)
        categories = data_library.get_categories()
        cats_by_id = {c['id']: c['name'] for c in categories}
        for l in labels:
            cat_name = cats_by_id[l['category_id']]
            cat_counts[cat_name] += 1

        artifact.metadata = {
            'n_examples': len(examples),
            'annotation_types': annotation_types,
            'category_counts': cat_counts,
        }

        with artifact.new_file(IMAGES_FNAME) as f:
            json.dump(examples, f, indent=2, sort_keys=True)
        with artifact.new_file(LABELS_FNAME) as f:
            json.dump(labels, f, indent=2, sort_keys=True)

    def __init__(self, examples, labels, artifact):
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
        return self.artifact().download()

    def artifact(self):
        """Return an artifact that represents this dataset."""
        return self._artifact
