import os

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == '__main__':
    print('Resizing images to 512x512, 768x768, and 1024x1024')
    ROOT_DIR = '/home/mchobanyan/data/kaggle/hpa-single-cell/'
    IMG_DIR = os.path.join(ROOT_DIR, 'train')
    OUT_DIR = os.path.join(ROOT_DIR, 'images')
    for filename in tqdm(os.listdir(IMG_DIR)):
        img = Image.open(os.path.join(IMG_DIR, filename))
        img.resize((512, 512)).save(os.path.join(OUT_DIR, '512x512', filename))
        img.resize((768, 768)).save(os.path.join(OUT_DIR, '768x768', filename))
        img.resize((1024, 1024)).save(os.path.join(OUT_DIR, '1024x1024', filename))
