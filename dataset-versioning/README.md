# Dataset versioning with W&B Artifacts

*W&B Artifacts is currently in the pilot phase, as we work with customers to refine the APIs and experience.*

This folder contains an end-to-end example of how to version your datasets using W&B Artifacts.

Please refer to our [Guide to Managing ML Datasets with W&B Artifacts](https://paper.dropbox.com/doc/Guide-Managing-ML-datasets-with-WB-Artifacts-draft-AmoGgY4qwdu4z2w7xHCMu) doc for an overview of the concepts used here.


## Running the example

Download example data.

```
sh download_coco_val.sh
```
Install requirements and setup.
```
pip install -r requirements.txt
python demo_setup.py
```

Add data to your data library.

```
python sim_collection_run.py col1 0 100000
python sim_collection_run.py col2 100000 200000
```

Initialize a wandb project in this folder.

```
$ wandb init
```

Create a new dataset artifact from a subset of your data library.

```
python create_dataset.py \
  --supercategories vehicle \
  --annotation_types bbox \
  --dataset_name "vehicle boxes"
```

Train a model based on the dataset.

```
python train.py \
  --dataset="vehicle boxes:latest" \
  --model_type=bbox
```

Create more dataset artifacts for other kinds of datasets.

```
sh create_many_datasets.sh
```

Update some labels in your data library.

```
python sim_modify_labels.py demodata/new_boxes.json
```

Create new dataset artifact versions for datasets that have updated labels.

```
python update_all_dataset_labels.py
```

Train a model on one of our updated datasets.

```
python train.py \
  --dataset="furniture boxes:latest" \
  --model_type=bbox
```
