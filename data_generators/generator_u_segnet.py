"""Input pipeline for the U-Seg-Net dataset.

The filenames have format "{id}.png".
"""
import os
import sys

sys.path.extend(['..'])

import numpy as np
import tensorflow as tf

from utils.utils import get_args
from utils.config import process_config

appendix = '.png'


class USegNetLoader:
    """
    Class that provides dataloading for U-SegNet model.
    The dataset will be [bs, im, im, 1].
    Every image has it mask.
    """
    def __init__(self, config):
        self.config = config

        data_dir = os.path.join('..', 'data', 'u_segnet', 'READY_U_SEGNET')

        train_dir = os.path.join(data_dir, 'train')
        eval_dir = os.path.join(data_dir, 'dev')
        test_dir = os.path.join(data_dir, 'test')

        # Get the file names from the train and dev sets
        self.train_images = np.array([os.path.join(train_dir, 'images', f) for f in os.listdir(os.path.join(train_dir, 'images')) if f.endswith(appendix)])
        self.eval_images  = np.array([os.path.join(eval_dir, 'images', f)  for f in os.listdir(os.path.join(eval_dir, 'images'))  if f.endswith(appendix)])
        self.test_images  = np.array([os.path.join(test_dir, 'images', f)  for f in os.listdir(os.path.join(test_dir, 'images'))  if f.endswith(appendix)])

        self.train_masks = np.array([os.path.join(train_dir, 'masks', f) for f in os.listdir(os.path.join(train_dir, 'masks')) if f.endswith(appendix)])
        self.eval_masks  = np.array([os.path.join(eval_dir, 'masks', f)  for f in os.listdir(os.path.join(eval_dir, 'masks'))  if f.endswith(appendix)])
        self.test_masks  = np.array([os.path.join(test_dir, 'masks', f)  for f in os.listdir(os.path.join(test_dir, 'masks'))  if f.endswith(appendix)])

        assert self.train_images.shape[0] == self.train_masks.shape[0], "Training files must have the same length"
        assert self.eval_images.shape[0] == self.eval_masks.shape[0], "Evaluation files must have the same length"

        # Define datasets sizes
        self.train_size = self.train_images.shape[0]
        self.eval_size = self.eval_images.shape[0]
        self.test_size = self.test_images.shape[0]

        # Define number of iterations per epoch
        self.num_iterations_train = (self.train_size + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_eval  = (self.eval_size  + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_test  = (self.test_size  + self.config.batch_size - 1) // self.config.batch_size

        self.features_placeholder = None
        self.labels_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        self._build_dataset_api()

    @staticmethod
    def _parse_function(filename, label, size):
        """Obtain the image and mask from the filename (for both training and validation).

        The following operations are applied:
            - Decode the image and mask from jpeg format
            - Convert to float and to range [0, 1]
        """
        image_string = tf.read_file(filename)
        mask_string = tf.read_file(label)

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        mask_decoded = tf.image.decode_jpeg(mask_string, channels=1)

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image_decoded, tf.float32)
        mask = tf.image.convert_image_dtype(mask_decoded, tf.float32)

        image = tf.image.resize_images(image, [size, size])
        mask = tf.image.resize_images(mask, [size, size])

        return image, mask

    @staticmethod
    def _train_preprocess(image, mask, use_random_flip, use_random_crop, crop_factor, mode='train'):
        """Image preprocessing for training.

        Apply the following operations:
            - Horizontally flip the image with probability 1/2
            - Apply random brightness and saturation
        """
        if mode == 'train':
            seed = np.random.randint(1234)
            if use_random_flip:
                image = tf.image.random_flip_left_right(image, seed)
                mask = tf.image.random_flip_left_right(mask, seed)

            if use_random_crop:
                image_size = image.shape
                crop_size = [crop_factor * image_size[0], crop_factor * image_size[1], 1]
                image = tf.image.resize_images(tf.image.random_crop(image, crop_size, seed), image_size)
                mask = tf.image.resize_images(tf.image.random_crop(mask, crop_size, seed), image_size)

            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

            # Make sure the image is still in [0, 1]
            image = tf.clip_by_value(image, 0.0, 1.0)

        return image, mask

    def _build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.features_placeholder = tf.placeholder(tf.string, [None, ])
            self.labels_placeholder = tf.placeholder(tf.string, [None, ])
            self.mode_placeholder = tf.placeholder(tf.string, shape=())

            # Create a Dataset serving batches of images and labels
            # We don't repeat for multiple epochs because we always train and evaluate for one epoch
            parse_fn = lambda f, l: self._parse_function(
                f,
                l,
                self.config.image_size
            )
            train_fn = lambda f, l: self._train_preprocess(
                f,
                l,
                self.config.use_random_flip,
                self.config.use_random_crop,
                self.config.crop_factor,
                self.mode_placeholder
            )

            self.dataset = (tf.data.Dataset.from_tensor_slices(
                (self.features_placeholder, self.labels_placeholder))
                .map(parse_fn, num_parallel_calls=self.config.num_parallel_calls)
                .map(train_fn, num_parallel_calls=self.config.num_parallel_calls)
                .batch(self.config.batch_size)
                .prefetch(1)  # make sure you always have one batch ready to serve
            )

            # Create reinitializable iterator from dataset
            self.iterator = self.dataset.make_initializable_iterator()

            self.iterator_init_op = self.iterator.initializer

            self.next_batch = self.iterator.get_next()

    def initialize(self, sess, mode='train'):
        if mode == 'train':
            idx = np.array(range(self.train_size))
            np.random.shuffle(idx)

            self.train_images = self.train_images[idx]
            self.train_masks  = self.train_masks[idx]

            sess.run(self.iterator_init_op, feed_dict={
                self.features_placeholder: self.train_images,
                self.labels_placeholder: self.train_masks,
                self.mode_placeholder: mode})

        elif mode == 'eval':
            sess.run(self.iterator_init_op, feed_dict={
                self.features_placeholder: self.eval_images,
                self.labels_placeholder: self.eval_masks,
                self.mode_placeholder: mode})

        else:
            sess.run(self.iterator_init_op, feed_dict={
                self.features_placeholder: self.eval_images,
                self.labels_placeholder: self.eval_masks,
                self.mode_placeholder: mode})

    def get_inputs(self):
        return self.next_batch


def main(config):
    """
    Function to test from console
    :param config:
    :return:
    """
    tf.reset_default_graph()

    sess = tf.Session()

    data_loader = USegNetLoader(config)

    images, labels = data_loader.get_inputs()

    print('Train')
    data_loader.initialize(sess, mode='train')

    out_im, out_l = sess.run([images, labels])

    print(out_im.shape, out_im.dtype)
    print(out_l.shape, out_l.dtype)

    print('Eval')
    data_loader.initialize(sess, mode='eval')

    out_im, out_l = sess.run([images, labels])

    print(out_im.shape, out_im.dtype)
    print(out_l.shape, out_l.dtype)

    print('Test')
    data_loader.initialize(sess, mode='test')

    out_im, out_l = sess.run([images, labels])

    print(out_im.shape, out_im.dtype)
    print(out_l.shape, out_l.dtype)


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print('Missing or invalid arguments %s' % e)
