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

    # Load in the image data from COCO JSON
    dataset = coco.COCO(args.json_file)
    images = dataset.loadImgs(dataset.getImgIds())
    cats = sorted(dataset.loadCats(dataset.getCatIds()), key=lambda x: x['id'])

    # Create a run, an artfact, a class object, and a table
    run = wandb.init(job_type='create-dataset')
    art = wandb.Artifact(type='dataset', name=args.name)
    class_set = wandb.Classes([{'name': c['name'], 'id': c['id']} for c in cats])
    table = wandb.Table(['image_id', 'ground_truth'])

    # Fill the table with images annotated with the box metdata
    for image in images:

        # Create the box metadata from COCO data
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

        # Create an image pointing to the correct file, annotated with box and class data
        wandb_image = wandb.Image(
            os.path.join(args.image_dir, image['file_name']),
            classes=class_set,
            boxes={
                'ground_truth': {
                    'box_data': boxes
                }
            })

        # Add a row to the table
        table.add_data(image['id'], wandb_image)

    # Add the table to the artifact
    art.add(table, 'data_table')

    # Log the artifact for future use
    run.log_artifact(art)

if __name__ == '__main__':
    main(sys.argv)
