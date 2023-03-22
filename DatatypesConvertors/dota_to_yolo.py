r"""
Script of a convertor for DOTA dataset to YOLO dataset.

Examples:
If the DOTA dataset has the following structure:
| DOTADataset
    | train
        | images
            | 001.png
            | 002.png
            | ...
            | 999.png
        | labelTxt
            | 001.txt
            | 002.txt
            | ...
            | 999.txt
    | val
        | images
            | 001.png
            | 002.png
            | ...
            | 999.png
        | labelTxt
            | 001.txt
            | 002.txt
            | ...
            | 999.txt

You need to type that command to get your YOLO dataset:
python dota_to_yolo.py --dota-path=your\dota\dataset\path --yolo-path=\your\new\yolo\dataset\path

The YOLO dataset will have the following structure, the only thing that changes is the way we annotate data:
| YOLODataset
    | train
        | images
            | 001.png
            | 002.png
            | ...
            | 999.png
        | labels
            | 001.txt
            | 002.txt
            | ...
            | 999.txt
    | val
        | images
            | 001.png
            | 002.png
            | ...
            | 999.png
        | labels
            | 001.txt
            | 002.txt
            | ...
            | 999.txt
"""



import json
import os
import glob
import shutil
import datetime
from shapely import Polygon

e = datetime.datetime.now()

CLASSES = ['wind_turbine']
IMG_HEIGHT = 512
IMG_WIDTH = 512

def get_categoryID(categoriesList, category):
    for cat in categoriesList:
        if cat["name"] == category:
            return cat["id"]
    return None

class DOTAtoYOLO():
    """
    Construct a data convertor that can create a dataset for object detection with YOLO format annotations based 
    on a dataset with DOTA format annotations.
    Args:
        path_to_DOTA (`str`):
            Path to the dataset with DOTA format annotations.
        path_to_COCO (`str`):
            Path to the new dataset with YOLO format annotations.
        labels (`list`):
            List of labels registered in annotations.
        height (`int`):
            Height of the images in DOTA format dataset.
        width (`int`):
            Width of the images in DOTA format dataset.
    """


    def __init__(
        self, 
        path_to_DOTA, 
        path_to_YOLO, 
        labels, 
        height, 
        width
    ):

        self.YOLOpath = path_to_YOLO
        self.DOTApath = path_to_DOTA

        self.dotaLabels = labels
        self.imagesHeight = height
        self.imagesWidth = width

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

    def process(self):
        categories = self.get_categories()
        for txtPath in glob.glob(os.path.join(self.DOTApath, '*', '*.txt')):
            imgPath = os.path.splitext(txtPath)[0] + '.png'
            imgName = imgPath.split("/")[-1]
            imgPath = os.path.join(self.DOTApath, 'images', imgName)
            with open(txtPath) as f:
                lines = f.readlines()
            yoloLines = []
            for line in lines:
                # We don't take the last element because it's just the "\n"
                line = line.split(' ')[:-1]
                bbox = [[float(line[i]), float(line[i+1])] for i in [0, 2, 4, 6]]
                [x, y] = list(zip(*bbox))
                minx, miny, maxx, maxy = min(x), min(y), max(x), max(y)
                x_c, y_c, w, h = (minx + maxx)/2, (miny + maxy)/2, maxx - minx, maxy - miny
                x_c, y_c, w, h = x_c/IMG_WIDTH, y_c/IMG_HEIGHT, w/IMG_WIDTH, h/IMG_HEIGHT
                category = line[-2]
                categoryID = get_categoryID(categories, category)
                yoloLine = f'{categoryID} {x_c} {y_c} {w} {h} \n'
                yoloLines.append(yoloLine)
            destinationTxtPath = os.path.join(self.YOLOpath, 'labels', txtPath.split("/")[-1])
            destinationImgPath = os.path.join(self.YOLOpath, 'images', imgPath.split("/")[-1])
            with open(destinationTxtPath, "w") as fp:
                fp.writelines(yoloLines)
            shutil.copy(imgPath, destinationImgPath)
    

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert DOTA dataset to YOLO dataset')
    parser.add_argument('--dota-path', help='Path to DOTA dataset')
    parser.add_argument('--yolo-path', help='Path to YOLO dataset')
    args = parser.parse_args()

    return args

args = parse_args()

os.makedirs(os.path.join(args.yolo_path, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(args.yolo_path, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(args.yolo_path, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(args.yolo_path, 'val', 'labels'), exist_ok=True)
print('Converting train dataset...')
TrainConvertor = DOTAtoYOLO(os.path.join(args.dota_path, 'train'), os.path.join(args.yolo_path, 'train'), CLASSES, IMG_HEIGHT, IMG_WIDTH)
TrainConvertor.process()
print('Converting val dataset...')
ValConvertor = DOTAtoYOLO(os.path.join(args.dota_path, 'val'), os.path.join(args.yolo_path, 'val'), CLASSES, IMG_HEIGHT, IMG_WIDTH)
ValConvertor.process()
