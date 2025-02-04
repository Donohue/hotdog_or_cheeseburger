#!/usr/bin/env python

from fastai.interpret import ClassificationInterpretation
from fastai.learner import load_learner
from fastai.vision.learner import vision_learner
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
import sys

import dataset

def predict(model_path, image_path):
    categories = ('cheeseburger', 'hotdog')
    learner = load_learner(model_path)
    img = Image.open(image_path)
    img.thumbnail((192,192))
    pred, idx, probs = learner.predict(img)
    category_prob_map = dict(zip(categories, map(float, probs)))
    for category, prob in category_prob_map.items():
        print(f'{category}: {prob*100:.2f}%')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s <model_path> <image_path>' % (sys.argv[0]))
        sys.exit(-1)
    predict(Path(sys.argv[1]), sys.argv[2])
