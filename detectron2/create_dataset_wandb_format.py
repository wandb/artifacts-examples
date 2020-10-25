import argparse
import os
import sys
import coco
import json
import wandb


parser = argparse.ArgumentParser(
    description='Create a dataset artifact using the COCO format.')
parser.add_argument('name', type=str,
                    help='Name for this dataset')
parser.add_argument('json_file', type=str,
                    help='Path to COCO dataset json')
parser.add_argument('image_dir', type=str,
                    help='Path to images directory, for images referenced in JSON')


def main(argv):
    args = parser.parse_args()

    dataset = coco.COCO(args.json_file)

    images = dataset.loadImgs(dataset.getImgIds())
    if len(images) == 0:
        raise ValueError('No images in dataset!')

    cats = sorted(dataset.loadCats(dataset.getCatIds()), key=lambda x: x['id'])

    run = wandb.init(job_type='create-dataset')

    art = wandb.Artifact(
        type='dataset',
        name=args.name,
        metadata={'format': {
            'type': 'wandb-table',
            'path': 'dataset.table.json',
            'column': 'ground_truth'
        }})

    class_set = wandb.Classes(
        [{'name': c['name'], 'id': c['id']} for c in cats])
    art.add(class_set, 'classes.json')

    table = wandb.Table(['ground_truth'])
    for image in images:
        image_path = os.path.join(args.image_dir, image['file_name'])
        anns = dataset.loadAnns(dataset.getAnnIds(imgIds=[image['id']]))
        boxes = []
        for ann in anns:
            box = ann['bbox']
            boxes.append({
                'domain': 'pixel',
                'position': {
                    'minX': box[0],
                    'maxX': box[0] + box[2],
                    'minY': box[1],
                    'maxY': box[1] + box[3]
                },
                'class_id': ann['category_id']
            })
        wandb_image = wandb.Image(image_path,
                                  boxes={
                                      'ground_truth': {
                                          'box_data': boxes
                                      }
                                  },
                                  classes={
                                      'path': 'classes.json'
                                  })
        table.add_data(wandb_image)
    art.add(table, 'dataset.table.json')

    run.log_artifact(art)


if __name__ == '__main__':
    main(sys.argv)
