set -e

python create_dataset.py \
  --supercategories vehicle \
  --annotation_types bbox \
  --dataset_name "vehicle-boxes-small" \
  --select_fraction 0.1

python create_dataset.py \
  --supercategories appliance \
  --annotation_types bbox \
  --dataset_name "appliance-boxes"

python create_dataset.py \
  --supercategories food \
  --annotation_types segmentation \
  --dataset_name "food-segmentation"

python create_dataset.py \
  --supercategories furniture \
  --annotation_types bbox \
  --dataset_name "furniture-boxes"

python create_dataset.py \
  --supercategories animal \
  --annotation_types bbox \
  --dataset_name "animal-boxes"

python create_dataset.py \
  --supercategories person \
  --annotation_types bbox \
  --dataset_name "people-boxes-small" \
  --select_fraction 0.1

python create_dataset.py \
  --supercategories person \
  --annotation_types bbox \
  --dataset_name "people-boxes-medium" \
  --select_fraction 0.3

python create_dataset.py \
  --supercategories person \
  --annotation_types bbox \
  --dataset_name "people-boxes-large-"

python create_dataset.py \
  --supercategories food \
  --categories "traffic light" "car" \
  --annotation_types bbox \
  --dataset_name "traffic-lights-cars-bounding-boxes-sampled" \
  --select_fraction 0.1
