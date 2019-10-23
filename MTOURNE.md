
TODO
====

check how the split is done in tf records

split 
======

Taco().load_dataset() expects annotations_<int>_<test/train/val>.json

to get it you can run :
`python split_dataset.py --dataset_dir /opt/data/TACO/data --nr_trials 1 --test_percentage 0`
(yolo train only uses a val, no test).

Note: the split could be done possibly using the tf records later.

class maps
==========
taco_configs/map_3.csv seems to be the one used for training
(turns off a lot of the smaller categories, or things like cigarette butt.)

from detector.py :
```
    with open(args.class_map) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]: row[1] for row in reader}
```

converting into tf records 
==========================

this is how pascal VOC dataset does it : https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py