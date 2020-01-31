"""
Run this file having split the annotations with 

python split_dataset.py --dataset_dir /opt/data/TACO/data --nr_trials 1 --test_percentage 0

this will generate two tf_dataset_*_.bin files, that are meant to be consume
by yolo-tf
"""

import csv

import dataset_tfrecord

class_map_filename = "/opt/data/TACO/detector/taco_config/map_3.csv"
with open(class_map_filename) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]: row[1] for row in reader}

dataset = dataset_tfrecord.TacoTF()
dataset.taco_to_tfrecord("/opt/data/TACO/data", 0, "train", "tf_dataset", class_map=class_map)
dataset.taco_to_tfrecord("/opt/data/TACO/data", 0, "val", "tf_dataset", class_map=class_map)
