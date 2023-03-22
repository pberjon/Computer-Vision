r"""
Script of a convertor for DOTA dataset to COCO dataset.

Examples:
If the DOTA dataset has the following structure:
| DOTADataset
    | train
        | images
            | 001.png
            | 002.png
            | ...
            | 800.png
        | labelTxt
            | 001.txt
            | 002.txt
            | ...
            | 800.txt
    | val
        | images
            | 801.png
            | 802.png
            | ...
            | 999.png
        | labelTxt
            | 801.txt
            | 802.txt
            | ...
            | 999.txt

You need to type that command to get your COCO dataset:
python dota_to_coco.py --dota-path=your\dota\dataset\path --coco-path=\your\new\coco\dataset\path

The COCO dataset will have the following structure:
| COCODataset
    | train
        | images
            | 001.png
            | 002.png
            | ...
            | 800.png
        | annotation.json
    | val
        | images
            | 801.png
            | 802.png
            | ...
            | 999.png
        | annotation.json
"""



import json
import os
import shutil
import datetime
import shapely

e = datetime.datetime.now()

CLASSES = ['wind_turbine']
IMG_HEIGHT = 512
IMG_WIDTH = 512

year = "2023"
version = "1.0"
description = "COCO annotation file for Wind Turbines Dataset"
contributor = "Pierre Berjon"
url = "github.com"
date_created = "%s-%s-%sT%s:%s:%s" % (e.year, e.month, e.day, e.hour, e.minute, e.second)

def get_categoryID(categoriesList, category):
    for cat in categoriesList:
        if cat["name"] == category:
            return cat["id"]
    return None

def get_imageID(imagesList, filename):
    for image in imagesList:
        if image["file_name"] == filename:
            return image["id"]
    return None

class DOTAtoCOCO():
    """
    Construct a data convertor that can create a dataset for object detection with COCO format annotations based 
    on a dataset with DOTA format annotations.
    Args:
        path_to_DOTA (`str`):
            Path to the dataset with DOTA format annotations.
        path_to_COCO (`str`):
            Path to the new dataset with COCO format annotations.
        labels (`tuple`):
            List of labels registered in annotations.
        height (`int`):
            Height of the images in DOTA format dataset.
        width (`int`):
            Width of the images in DOTA format dataset.
    """


    def __init__(
        self, 
        path_to_DOTA, 
        path_to_COCO, 
        labels, 
        height, 
        width
    ):

        self.COCOpath = path_to_COCO
        self.DOTApath = path_to_DOTA

        self.dotaLabels = labels
        self.imagesHeight = height
        self.imagesWidth = width
        
    
    def get_info(self):
        return {
            "year": year,
            "version": version,
            "description": description,
            "contributor": contributor,
            "url": url,
            "date_created": date_created
        }
    
    def get_categories(self):
        labels = self.dotaLabels
        categories = []
        for i in range(len(labels)):
            label = labels[i]
            categories.append({
                "id": i,
                "name": label
            })
        
        return categories
    
    def get_images(self):
        images = []
        imagesFiles = os.listdir(os.path.join(self.DOTApath, 'images'))
        for idx in range(len(imagesFiles)):
            imageFile = imagesFiles[idx]
            image = {
                "id": idx,
                "license": 1,
                "file_name": os.path.join('images', imageFile),
                "height": self.imagesHeight,
                "width": self.imagesWidth,
                "date_captured": None
            }
            images.append(image)
        
        return images

    def get_annotations(self):
        print('Annotations processing...')
        annotations = []
        labelsFiles = os.listdir(os.path.join(self.DOTApath, 'labelTxt'))
        images = self.get_images()
        categories = self.get_categories()
        record_id = 0
        for idx in range(len(labelsFiles)):
            labelFile = labelsFiles[idx]
            labelName, labelExtension = os.path.splitext(labelFile)
            imageID = get_imageID(images, os.path.join('images', labelName + '.png'))
            with open(os.path.join(self.DOTApath, 'labelTxt', labelFile)) as f:
                lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                line = line.split(' ')
                coords = line[:-3]
                bbox = [[float(coords[i]), float(coords[i+1])] for i in [0, 2, 4, 6]]
                [x, y] = list(zip(*bbox))
                minx, miny, maxx, maxy = min(x), min(y), max(x), max(y)
                bbox = shapely.geometry.box(minx, miny, maxx, maxy)
                area = bbox.area
                bbox = [(minx + maxx)/2, (miny + maxy)/2, maxx - minx, maxy - miny]
                category = line[-3]
                label = {
                    "id": record_id,
                    "image_id": imageID,
                    "category_id": get_categoryID(categories, category),
                    "bbox": bbox,
                    "area": area
                }

                record_id += 1

                annotations.append(label)
        return annotations

    def copy_images(self):
        print('Copying all the images into images folder...\n')
        for imageFile in os.listdir(os.path.join(self.DOTApath, 'images')):
            shutil.copy(
                os.path.join(self.DOTApath, 'images', imageFile),
                os.path.join(self.COCOpath, 'images', imageFile)
            )

    def process(self):
        info = self.get_info()
        categories = self.get_categories()
        images = self.get_images()
        annotations = self.get_annotations()
        coco_data = CocoAnnotation(info, categories, images, annotations)

        coco_data.save_annotation(os.path.join(self.COCOpath, 'annotation.json'))
        self.copy_images()

class CocoAnnotation():
    def __init__(
        self, 
        info, 
        categories, 
        images, 
        annotations
    ):

        self.info = info
        self.categories = categories
        self.images = images
        self.annotations = annotations
    
    def get_annotation(self):
        ann = {}
        ann["info"] = self.info
        ann["categories"] = self.categories
        ann["images"] = self.images
        ann["annotations"] = self.annotations

        return ann
    
    def save_annotation(self, destination_path):
        print('Saving the annotation file...')
        ann = self.get_annotation()
        with open(destination_path, "w") as fp:
            json.dump(ann, fp)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert DOTA dataset to COCO dataset')
    parser.add_argument('--dota-path', help='Path to DOTA dataset')
    parser.add_argument('--coco-path', help='Path to COCO dataset')
    args = parser.parse_args()

    return args

args = parse_args()

os.makedirs(os.path.join(args.coco_path, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(args.coco_path, 'val', 'images'), exist_ok=True)
print('Converting train dataset...')
TrainConvertor = DOTAtoCOCO(os.path.join(args.dota_path, 'train'), os.path.join(args.coco_path, 'train'), CLASSES, IMG_HEIGHT, IMG_WIDTH)
TrainConvertor.process()
print('Converting val dataset...')
ValConvertor = DOTAtoCOCO(os.path.join(args.dota_path, 'val'), os.path.join(args.coco_path, 'val'), CLASSES, IMG_HEIGHT, IMG_WIDTH)
ValConvertor.process()
