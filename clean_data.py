#!/usr/bin/env python

from pathlib import Path
from fastai.interpret import ClassificationInterpretation
from fastai.learner import load_learner
from fastai.vision.learner import vision_learner
from matplotlib import pyplot as plt
import sys

import dataset

def clean_data(model_path):
    learner = load_learner(model_path)
    learner.dls = dataset.dataloaders()
    learner.eval()

    interp = ClassificationInterpretation.from_learner(learner)
    interp.plot_confusion_matrix()
    interp.plot_top_losses(5, nrows=1)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <model_path>' % sys.argv[0])
        sys.exit(-1)
    clean_data(Path(sys.argv[1]))
