import argparse
import json
import os
import sys
import coco
import random
import wandb

parser = argparse.ArgumentParser(
    description='Produce a subset of a coco formatted dataset')
parser.add_argument('json_file', type=str,
                    help='Path to COCO dataset json')
parser.add_argument('output_json_file', type=str,
                    help='Path to write output dataset to')

parser.add_argument('--supercategories', type=str, nargs='*', default=[],
                    help='coco supercategories to take examples from')
parser.add_argument('--categories', type=str, nargs='*', default=[],
                    help='coco categories to take examples from')
parser.add_argument('--select_fraction', type=float, default=1,
                    help='random fraction of examples to select')
parser.add_argument('--after_fraction', type=float, default=0,
                    help='random fraction of examples to select')

def main(argv):
    args = parser.parse_args()

    dataset = coco.COCO(args.json_file)

    if args.supercategories or args.categories:
        cat_ids = dataset.getCatIds(
            catNms=args.categories, supNms=args.supercategories)
    else:
        cat_ids = dataset.getCatIds()

    # union of images that contain any of cat_ids
    image_ids = set()
    for cat_id in cat_ids:
        image_ids |= set(dataset.getImgIds(catIds=cat_id))
    start_idx = int(args.after_fraction * len(image_ids))
    end_idx = start_idx + int(args.select_fraction * len(image_ids))
    image_ids = sorted(image_ids)[start_idx:end_idx]

    ann_ids = dataset.getAnnIds(imgIds=image_ids, catIds=cat_ids)

    result_json = {
        'info': {},
        'licenses': dataset.dataset['licenses'],
        'images': sorted(dataset.loadImgs(image_ids), key=lambda x: x['id']),
        'annotations': sorted(dataset.loadAnns(ann_ids), key=lambda x: x['id']),
        'categories': sorted(dataset.loadCats(dataset.getCatIds()), key=lambda x: x['id'])
    }

    json.dump(result_json, open(args.output_json_file, 'w'), sort_keys=True)


if __name__ == '__main__':
    main(sys.argv)