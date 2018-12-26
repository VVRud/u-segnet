from utils.utils import get_args
from utils.logger import DefinedSummarizer
from utils.dirs import create_dirs
from utils.config import process_config
from trainers.trainer_u_segnet import USegNetTrainer
from models.model_u_segnet import USegNetModel
from data_generators.generator_u_segnet import USegNetLoader
import tensorflow as tf
import sys

sys.path.extend(['..'])


def main():

    try:
        args = get_args()
        config = process_config(args.config)

    except Exception as e:
        print("missing or invalid arguments")
        print(e)
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    sess = tf.Session(config=conf)

    with sess.as_default():

        data_loader = USegNetLoader(config)

        model = USegNetModel(data_loader, config)

        logger = DefinedSummarizer(
            sess,
            summary_dir=config.summary_dir,
            scalar_tags=[
                'train/loss_per_epoch',
                'train/dice_per_epoch',
                'train/iou_per_epoch',
                'eval/loss_per_epoch',
                'eval/dice_per_epoch',
                'eval/iou_per_epoch'
            ]
        )

        trainer = USegNetTrainer(sess, model, config, logger, data_loader)

        trainer.train()


if __name__ == '__main__':
    main()
