set -e

# setup
rm -rf artifact
rm -rf artifacts
rm -rf bucket
python demo_setup.py

# simulate uploading two collection runs to our data library
python sim_collection_run.py col1 0 100000
python sim_collection_run.py col2 100000 200000

# create a dataset from our data library
python create_dataset.py \
  --supercategories vehicle \
  --annotation_types bbox \
  --dataset_name "vehicle-boxes"

# train a model on the the uploaded dataset
python train.py \
  --dataset="vehicle-boxes:latest" \
  --model_type=bbox

sh create_many_datasets.sh

python sim_modify_labels.py demodata/new_boxes.json

python update_all_dataset_labels.py

python train.py \
  --dataset="furniture-boxes:latest" \
  --model_type=bbox