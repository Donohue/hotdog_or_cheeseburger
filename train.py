#!/usr/bin/env python

from pathlib import Path
from fastai.metrics import error_rate
from fastai.callback.schedule import fine_tune
from fastai.vision.learner import vision_learner
from torchvision.models.resnet import resnet18
from matplotlib import pyplot as plt

import dataset

def train_model():
    dls = dataset.dataloaders()
    dls.show_batch(max_n=16)
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4) 
    learn.export('models/model.pth')
    plt.show()

if __name__ == '__main__':
    train_model()
