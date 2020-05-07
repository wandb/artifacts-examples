import argparse
import json
import sys

import data_library

parser = argparse.ArgumentParser(description='')
parser.add_argument('label_file', type=str, help='')

def main(argv):
    args = parser.parse_args()

    seg_labels = data_library.get_seg_labels()
    box_labels = data_library.get_box_labels()

    new_labels = json.load(open(args.label_file))

    for label in new_labels:
        label_id = label['id']
        if 'segmentation' in label:
            seg_labels[label_id] = label
        elif 'bbox' in label:
            box_labels[label_id] = label
    
    data_library.save_seg_labels(seg_labels)
    data_library.save_box_labels(box_labels)
        

if __name__ == '__main__':
    main(sys.argv)