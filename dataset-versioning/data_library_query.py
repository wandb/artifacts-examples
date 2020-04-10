import os
import tempfile
import json

import data_library

def categories_filtered(supercategories, categories):
    cats = data_library.get_categories()
    chosen_cats = [c for c in cats
        if c['supercategory'] in supercategories or
           c['name'] in categories]
    return chosen_cats

def labels_of_types(types):
    labels = []
    if 'bbox' in types:
        labels.extend(data_library.get_box_labels().values())
    if 'segmentation' in types:
        labels.extend(data_library.get_seg_labels().values())
    return labels

def labels_of_types_and_categories(types, category_ids):
    type_labels = labels_of_types(types)
    result = []
    for label in type_labels:
        if label['category_id'] in category_ids:
            result.append(label)
    return result

def labels_for_images(image_paths, label_types):
    labels = labels_of_types(label_types)
    return [l for l in labels if l['image_path'] in image_paths]