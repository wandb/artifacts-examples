import os
import tempfile
import json
import sys

import bucket_api

CATEGORIES_PATH = 'labels/categories.json'
BBOX_PATH = 'labels/bounding_boxes.json'
SEGMENTATION_PATH = 'labels/segmentation.json'

def get_image_path(collection_id, example_id):
    return '/'.join(('collections', collection_id, '%012u.jpg' % example_id))

def get_categories():
    bucketapi = bucket_api.get_bucket_api()
    with tempfile.TemporaryDirectory() as tempdir:
        local_path = os.path.join(tempdir, 'categories.json')
        exists = bucketapi.download_file(CATEGORIES_PATH, local_path)
        if exists:
            return json.load(open(local_path))
        else:
            print('You must run demo_setup.py first!')
            sys.exit(1)

def get_labels(bucket_path):
    bucketapi = bucket_api.get_bucket_api()
    with tempfile.TemporaryDirectory() as tempdir:
        local_path = os.path.join(tempdir, 'labels.json')
        exists = bucketapi.download_file(bucket_path, local_path)
        if exists:
            labels = json.load(open(local_path))
            labels_by_id = {}
            for label in labels:
                labels_by_id[label['id']] = label
        else:
            labels_by_id = {}
    return labels_by_id

def get_box_labels():
    return get_labels(BBOX_PATH)

def get_seg_labels():
    return get_labels(SEGMENTATION_PATH)

def save_labels(bucket_path, labels_by_id):
    bucketapi = bucket_api.get_bucket_api()
    with tempfile.TemporaryDirectory() as tempdir:
        local_path = os.path.join(tempdir, 'labels.json')
        json.dump(
            sorted(labels_by_id.values(), key=lambda a: a['id']),
            open(local_path, 'w'),
            sort_keys=True)
        bucketapi.upload_file(bucket_path, local_path)

def save_box_labels(labels_by_id):
    return save_labels(BBOX_PATH, labels_by_id)

def save_seg_labels(labels_by_id):
    return save_labels(SEGMENTATION_PATH, labels_by_id)