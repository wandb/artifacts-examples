import argparse
import os
import sys
import coco
import json
import wandb


parser = argparse.ArgumentParser(
    description='Create a dataset artifact using the COCO format.')
parser.add_argument('--name', type=str,
                    help='Name for this dataset')
parser.add_argument('--json_file', type=str,
                    help='Path to COCO dataset json')
parser.add_argument('--image_dir', type=str,
                    help='Path to images directory, for images referenced in JSON')
parser.add_argument('--supercategories', type=str, nargs='*', default=[],
                    help='coco supercategories to take examples from')
parser.add_argument('--categories', type=str, nargs='*', default=[],
                    help='coco categories to take examples from')
parser.add_argument('--select_fraction', type=float, default=1,
                    help='random fraction of examples to select')
parser.add_argument('--after_fraction', type=float, default=0,
                    help='random fraction of examples to select')

def make_coco_subset_json(args, output_json_file):
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
    json.dump(result_json, open(output_json_file, 'w'), sort_keys=True)

def main(argv):
    args = parser.parse_args()
    
    # Load in the image data from COCO JSON. This section downsamples the data
    # according to the arguments and creates some additional metadata & objects
    # used in the code below
    subset_json_path = "subset_coco.json"
    make_coco_subset_json(args, subset_json_path)
    dataset = coco.COCO(subset_json_path)
    images = dataset.loadImgs(dataset.getImgIds())
    cats = sorted(dataset.loadCats(dataset.getCatIds()), key=lambda x: x['id'])
    thing_ids = [k["id"] for k in cats]
    # Warning, integer keys get converted to strings when json serialized.
    # We fix this up at read time.
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in cats]
    detectron2_metadata = {
        'thing_dataset_id_to_contiguous_id': thing_dataset_id_to_contiguous_id,
        'thing_classes': thing_classes,
    }

    # Create a run, an artfact, a class object, and a table
    run = wandb.init(job_type='create-dataset')
    art = wandb.Artifact(type='dataset', name=args.name)
    with art.new_file('detectron2_metadata.json') as f:
        json.dump(detectron2_metadata, f)
    class_set = wandb.Classes([{'name': c['name'], 'id': c['id']} for c in cats])
    columns = list(images[0].keys())
    table = wandb.Table(columns = columns + ["image"])

    # Fill the table with images annotated with the box metdata
    for image in images:
        
        # Add the image file directly (only required if we care about the path)
        image_path = os.path.join(args.image_dir, image['file_name'])
        artifact.add_file(image_path, name='/'.join(('images', image['file_name'])))

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
        table.add_data(*[
            image[key] for key in columns
        ] + [wandb_image])

    # Add the table to the artifact
    art.add(table, 'data_table')

    # Log the artifact for future use
    run.log_artifact(art)

if __name__ == '__main__':
    main(sys.argv)
