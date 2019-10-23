import json
import os 
import io
import hashlib

import dataset
import dataset_util

from pycocotools.coco import COCO
import tensorflow as tf
import PIL

class TacoTF(dataset.Taco):

    def taco_to_tfrecord(self, dataset_dir, round, subset, tf_output, class_ids=None,
                class_map=None, return_taco=False, auto_download=False):
        """Load a subset of the TACO dataset.
        AND convert it to TF record

        dataset_dir: The root directory of the TACO dataset.
        round: split number
        subset: which subset to load (train, val, test)
        class_ids: If provided, only loads images that have the given classes.
        class_map: Dictionary used to assign original classes to new class system
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if not tf_output:
            raise ValueError("please provide a tf_output prefix parameter")

        # TODO: Once we got the server running
        # if auto_download is True:
        #     self.auto_download(dataset_dir, subset, year)
        ann_filepath = os.path.join(dataset_dir , 'annotations_')
        if round != None:
            ann_filepath += str(round) + "_" + subset + ".json"
        else:
            ann_filepath += subset + ".json"

        assert os.path.isfile(ann_filepath)

        # Load dataset
        dataset = json.load(open(ann_filepath, 'r'))

        # Replace dataset original classes before calling the coco Constructor
        # Some classes may be assigned background to remove them from the dataset
        self.replace_dataset_classes(dataset, class_map)

        taco_alla_coco = COCO()
        taco_alla_coco.dataset = dataset
        taco_alla_coco.createIndex()

        # Add images and classes except Background
        # Definitely not the most efficient way
        image_ids = []
        background_id = -1
        class_ids = sorted(taco_alla_coco.getCatIds())
        for i in class_ids:
            class_name = taco_alla_coco.loadCats(i)[0]["name"]
            if class_name != 'Background':
                self.add_class("taco", i, class_name)
                image_ids.extend(list(taco_alla_coco.getImgIds(catIds=i)))
            else:
                background_id = i
        image_ids = list(set(image_ids))

        if background_id > -1:
            class_ids.remove(background_id)

        print('Number of images used:', len(image_ids))

        ## Write all the classes, ordered by id ##
        # include Background to class labels
        # just nothing gets trained on class_id = 0
        all_class_ids = sorted(taco_alla_coco.getCatIds())
        with open(tf_output + "_classes.names", "w") as f:
            for class_id in all_class_ids:
                name = taco_alla_coco.cats[class_id]['name']   
                f.write(name + "\n")

        writer = tf.io.TFRecordWriter(tf_output + "_" + subset + ".bin")

        # Add images
        for i in image_ids:
            
            height = taco_alla_coco.imgs[i]["height"]
            width = width=taco_alla_coco.imgs[i]["width"]
            img = taco_alla_coco.imgs[i]

            annotations = taco_alla_coco.imgToAnns[img['id']]
            
            ## lists of features per image ##
            # bbox coordinates
            xminl = []
            yminl = []
            xmaxl = []
            ymaxl = []
            # category (numeric)
            catl = []
            # category name
            labell = []

            # default stuff from pascal voc (always set empty here.)
            viewl = [] # I think the original voc contains stuff like "frontal"
            truncatedl = []
            difficultl = []

            for ann in annotations:
                if ann['category_id'] == 0:
                    # if a label category has been replaced by category_id == 0
                    # when loading the category map we skip it
                    # because it's Background. and we don't need to a learn a bbox
                    # with background.
                    continue

                # category
                catl.append(ann['category_id'])

                # label of the category
                cat = taco_alla_coco.cats[ann['category_id']]
                labell.append(cat['name'].encode('utf8'))

                # bbox
                xmin, ymin, bbox_width, bbox_height = tuple(ann['bbox'])
                xminl.append(float(xmin) / width)
                yminl.append(float(ymin) / height)
                xmaxl.append(float(xmin + bbox_width) / width)
                ymaxl.append(float(ymin + bbox_height) / height)

                # defaults
                viewl.append("".encode('utf8'))
                truncatedl.append(0)
                difficultl.append(0)
            
            filename = os.path.join(dataset_dir, taco_alla_coco.imgs[i]['file_name'])
            filename = filename.encode('utf8')
            with tf.io.gfile.GFile(filename, 'rb') as fid:
                encoded_jpg = fid.read()
                encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            key = hashlib.sha256(encoded_jpg).hexdigest()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xminl),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxl),
                'image/object/bbox/ymin': dataset_util.float_list_feature(yminl),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxl),
                'image/object/class/text': dataset_util.bytes_list_feature(labell),
                'image/object/class/label': dataset_util.int64_list_feature(catl),

                # we put these in just to be look like pascal voc example
                # but they're always set to defaults 
                # see : https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py#L124
                'image/object/difficult': dataset_util.int64_list_feature(difficultl),
                'image/object/truncated': dataset_util.int64_list_feature(truncatedl),
                'image/object/view': dataset_util.bytes_list_feature(viewl),
                }))
            writer.write(example.SerializeToString())

        writer.close()

