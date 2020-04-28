import argparse
from datetime import datetime
import glob
import json
import os
import sys

from demodata.coco import COCO

import bucket_api
import data_library

parser = argparse.ArgumentParser(description='')
parser.add_argument('collection_id', type=str, help='')
parser.add_argument('min_example_id', type=int, help='')
parser.add_argument('max_example_id', type=int, help='')


def main(argv):
    args = parser.parse_args()
    bucketapi = bucket_api.get_bucket_api()
    coco_api = COCO('./demodata/coco/annotations/instances_val2017.json')

    collection_id = args.collection_id

    img_ids = [id_ for id_ in coco_api.getImgIds()
        if id_ > args.min_example_id <= args.max_example_id]
    imgs = coco_api.loadImgs(img_ids)
    for img in imgs:
        example_id = img['id']
        fname = os.path.basename(img['coco_url'])
        bucketapi.upload_file(
            data_library.get_image_path(collection_id, example_id),
            os.path.join('demodata', 'coco', 'val2017', fname))
    
    ann_ids = coco_api.getAnnIds(imgIds=img_ids)
    annotations = coco_api.loadAnns(ann_ids)

    cur_box_annos = data_library.get_box_labels()

    cur_seg_annos = data_library.get_seg_labels()

    for anno in annotations:
        anno_id = anno['id']
        if 'segmentation' in anno:
            cur_seg_annos[anno_id] = {
                'id': anno_id,
                'segmentation': anno['segmentation'],
                'category_id': anno['category_id'],
                'image_path': data_library.get_image_path(collection_id, anno['image_id'])
            }
        if 'bbox' in anno:
            cur_box_annos[anno_id] = {
                'id': anno_id,
                'bbox': anno['bbox'],
                'category_id': anno['category_id'],
                'image_path': data_library.get_image_path(collection_id, anno['image_id'])
            }
        
    json.dump(
        sorted(cur_box_annos.values(), key=lambda a: a['id']),
        open('boxes.json', 'w'), sort_keys=True)
    bucketapi.upload_file(data_library.BBOX_PATH, 'boxes.json')
    json.dump(
        sorted(cur_seg_annos.values(), key=lambda a: a['id']),
        open('seg.json', 'w'), sort_keys=True)
    bucketapi.upload_file(data_library.SEGMENTATION_PATH, 'seg.json')


if __name__ == '__main__':
    main(sys.argv)
