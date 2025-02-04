#!/usr/bin/env python

import os
from pathlib import Path
from fastbook import search_images_ddg
from fastai.data.transforms import get_image_files
from fastai.vision.all import download_images
from fastai.vision.utils import verify_images

def generate_images(query):
    path = Path(query)
    path.mkdir(exist_ok=True)
    results = search_images_ddg(query)
    download_images(path, urls=results)
    files = get_image_files(path)
    failed = verify_images(files)
    failed.map(Path.unlink)

if __name__ == '__main__':
    queries = ['hotdog', 'cheeseburger']
    for query in queries:
        generate_images(query)
