#!/usr/bin/env python

from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import get_image_files, RandomSplitter, parent_label
from fastai.vision.augment import RandomResizedCrop, aug_transforms
from fastai.vision.data import ImageBlock


def get_image_files_in_folders(path):
    folders = ['hotdog', 'cheeseburger']
    return get_image_files(path, folders=folders)


def datablock():
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files_in_folders,
        splitter=RandomSplitter(valid_pct=0.2, seed=3),
        get_y=parent_label,
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms())

def dataloaders(path='.'):
	return datablock().dataloaders(path)
