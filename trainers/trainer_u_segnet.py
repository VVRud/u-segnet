from base.base_train import BaseTrain
from tqdm import tqdm

import tensorflow as tf

from utils.metrics import AverageMeter


class USegNetTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader):
        """
        Constructing the U-SegNet trainer based on the Base Train..
        Here is the pipeline of constructing
        - Assign sess, model, config, logger, data_generators(if_specified)
        - Initialize all variables
        - Load the latest checkpoint
        - Create the summarizer
        - Get the nodes we will need to run it from the graph
        :param sess:
        :param model:
        :param config:
        :param logger:
        :param data_loader:
        """
        super(USegNetTrainer, self).__init__(sess, model, config, logger, data_loader)

        self.model.load(self.sess)

        self.summarizer = logger

        self.x, self.y, self.is_training = tf.get_collection('inputs')
        self.train_op, self.loss_node, self.dice_node, self.iou_node = tf.get_collection('train')
        self.predictions = tf.get_collection('predictions')

    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        :return:
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.test(cur_epoch)

    def train_epoch(self, epoch=None):
        """
        Train one epoch.
        :param epoch: Current epoch number.
        :return:
        """
        self.data_loader.initialize(self.sess, mode='train')

        tt = tqdm(range(
            self.data_loader.num_iterations_train),
            total=self.data_loader.num_iterations_train,
            desc="epoch-{}-".format(epoch)
        )

        loss_per_epoch = AverageMeter()
        dice_per_epoch = AverageMeter()
        iou_per_epoch = AverageMeter()

        for cur_it in tt:
            loss, dice, iou = self.train_step()
            print(loss, dice, iou)

            loss_per_epoch.update(loss)
            dice_per_epoch.update(dice)
            iou_per_epoch.update(iou)

        self.sess.run(self.model.global_epoch_inc)

        summaries_dict = {
            'train/loss_per_epoch': loss_per_epoch.val,
            'train/dice_per_epoch': dice_per_epoch.val,
            'train/iou_per_epoch': iou_per_epoch.val
        }

        self.model.save(self.sess)

        print("""
Epoch-{} Train loss:{:.4f} -- dice:{:.4f} -- iou:{:.4f}
        """.format(epoch, loss_per_epoch.val, dice_per_epoch.val, iou_per_epoch.val))

        tt.close()

    def train_step(self):
        """
        Run the session of train_step in tensorflow
        also get the loss, dice and iou of that minibatch.
        :return: (loss, dice, iou) tuple of some metrics to be used in summaries
        """
        _, loss, dice, iou = self.sess.run(
            [
                self.train_op,
                self.loss_node,
                self.dice_node,
                self.iou_node
            ],
            feed_dict={self.is_training: True}
        )

        return loss, dice, iou

    def test(self, epoch):
        self.data_loader.initialize(self.sess, mode='eval')

        tt = tqdm(
            range(self.data_loader.num_iterations_test),
            total=self.data_loader.num_iterations_test,
            desc="Val-{}-".format(epoch)
        )

        loss_per_epoch = AverageMeter()
        dice_per_epoch = AverageMeter()
        iou_per_epoch = AverageMeter()

        for cur_it in tt:
            loss, dice, iou = self.sess.run(
                [
                    self.loss_node,
                    self.dice_node,
                    self.iou_node
                ],
                feed_dict={self.is_training: False}
            )

            loss_per_epoch.update(loss)
            dice_per_epoch.update(dice)
            iou_per_epoch.update(iou)

        summaries_dict = {
            'eval/loss_per_epoch': loss_per_epoch.val,
            'eval/dice_per_epoch': dice_per_epoch.val,
            'eval/iou_per_epoch': iou_per_epoch.val
        }

        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        print("""
Val-{} Eval loss:{:.4f} -- dice:{:.4f} -- iou:{:.4f}
        """.format(epoch, loss_per_epoch.val, dice_per_epoch.val, iou_per_epoch.val))

        tt.close()
