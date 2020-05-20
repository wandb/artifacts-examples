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

    # create detectron2 metadata
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

    run = wandb.init(job_type='create-dataset')

    artifact = wandb.Artifact(
        type='dataset',
        metadata={'format': {'type': 'coco_dataset_json'}})
    artifact.add_file(args.json_file, name='annotations.json')

    with artifact.new_file('detectron2_metadata.json') as f:
        json.dump(detectron2_metadata, f)

    for image in images:
        image_path = os.path.join(args.image_dir, image['file_name'])
        artifact.add_file(image_path, name='/'.join(('images', image['file_name'])))

    run.log_artifact(artifact, name=args.name)


if __name__ == '__main__':
    main(sys.argv)