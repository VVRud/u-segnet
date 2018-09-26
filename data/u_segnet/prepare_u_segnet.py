import os
import sys

sys.path.extend(['../..'])

import random
from PIL import Image
from tqdm import tqdm

from utils.utils import get_args
from utils.config import process_config


def resize_and_save(filename, output_dir, size):
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


def main(image_size, train_percentage):
    train_image_dir = os.path.join('train', 'images')
    test_data_dir = os.path.join('test', 'images')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_image_dir)
    filenames = [os.path.join(train_image_dir, f) for f in filenames if f.endswith('.png')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.png')]

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(train_percentage * len(filenames))
    train_filenames = filenames[:split]
    dev_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    # Define output dir
    output_dir = 'READY_U_SEGNET'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print('Warning: output dir {} already exists"'.format(output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(output_dir, split)

        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print('Warning: dir {} already exists'.format(output_dir_split))

        if not (os.path.exists(output_dir_split + '/images') or os.path.exists(output_dir_split + '/masks')):
            os.mkdir(output_dir_split + '/images')
            os.mkdir(output_dir_split + '/masks')
        else:
            print('Warning: dirs already exist')

        print('Processing {} data, saving preprocessed data to {}'.format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split + '/images', image_size)
            if split != 'test':
                resize_and_save(filename.replace('images', 'masks'), output_dir_split + '/masks', image_size)

    print('Done building dataset')

if __name__ == '__main__':
    try:
        args = get_args()
        config = process_config(args.config)
        main(config.image_size, config.train_percentage)

    except Exception as e:
        print('Missing or invalid arguments %s' % e)
        print('Using default 128x128 size and train percentage 80%')
        main(128, 0.8)
