import argparse
import torch
import os
import cv2
import json
import sys

from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

import wandb

parser = argparse.ArgumentParser(
    description='Create a dataset artifact using the COCO format.')
parser.add_argument('artifact_name', type=str,
                    help='Artifact containing predictions')

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'

def remove_prefix(from_string, prefix):
    return from_string[len(prefix):]

def main(argv):
    args = parser.parse_args()
    api = wandb.Api()

    artifact = api.artifact(type='result', name=args.artifact_name)

    dataset_json = json.load(open(artifact.get_path('dataset.json').download()))
    ds_uri = dataset_json['dataset_artifact'][0]
    if not ds_uri.startswith(WANDB_ARTIFACT_PREFIX):
        raise ValueError('can\'t load uri %s' % ds_uri)
    ds_artifact_name = remove_prefix(ds_uri, WANDB_ARTIFACT_PREFIX)
    ds_artifact = api.artifact(type='dataset', name=ds_artifact_name)

    path = artifact.get_path('instances_predictions.pth').download()

    preds = torch.load(path)

    for pred in preds:
        image_id = pred['image_id']

        # TODO: We're reconstructing the path. We should use detectron's dataset stuff
        # instead to make this generic
        im_path = ds_artifact.get_path(os.path.join('images', '%012u.jpg' % image_id)).download()

        im = cv2.imread(im_path)

        v = Visualizer(im[:, :, ::-1],
                    metadata={}, 
                    scale=0.8, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        # Draw the boxes directly, the instances_predictions.pth file is not
        # actually in the detectron2 prediction format. This seems like a detectron2
        # bug.
        for instance in pred['instances']:
            print('SCORE', instance['score'])
            if instance['score'] > .5:
                v.draw_box(instance['bbox'])
        cv2.imshow('image', v.output.get_image()[:, :, ::-1])
        cv2.waitKey()

if __name__ == '__main__':
    main(sys.argv)
