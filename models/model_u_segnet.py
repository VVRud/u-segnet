from base.base_model import BaseModel
import tensorflow as tf
import utils.metrics as mt
import numpy as np


class USegNetModel(BaseModel):
    """
    Implementation of the model proposed by IIIT-Delhi research team.
    https://arxiv.org/abs/1806.04429
    """
    def __init__(self, data_loader, config):
        super(USegNetModel, self).__init__(config)

        self.data_loader = data_loader

        self.x = None
        self.y = None
        self.is_training = None
        self.out_argmax = None

        self.loss = None
        self.dice = None
        self.optimizer = None
        self.train_step = None

        self.build_model()
        self.init_saver()

    def build_model(self):
        """
        Creates model

        :return:
        """

        """
        Helper Variables
        """
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        """
        Inputs to the network
        """
        with tf.variable_scope('inputs'):
            self.x, self.y = self.data_loader.get_inputs()

            assert self.x.get_shape().as_list() == [None, self.config.image_size, self.config.image_size, 1]

            self.is_training = tf.placeholder(tf.bool, name='Training_flag')

        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        """
        Network Architecture
        """
        out = self.x

        with tf.variable_scope('network'):
            """
            Encoder
            """
            out = self.conv_bn_relu(out, 64, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='conv1_1')
            conv1 = self.conv_bn_relu(out, 64, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='conv1_2')
            pool1, pool1_ind = self.pool(conv1, name='pool1')

            out = self.conv_bn_relu(pool1, 128, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='conv2_1')
            out = self.conv_bn_relu(out, 128, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='conv2_2')
            pool2, pool2_ind = self.pool(out, name='pool2')

            out = self.conv_bn_relu(pool2, 256, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='conv3_1')
            out = self.conv_bn_relu(out, 256, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='conv3_2')
            pool3, pool3_ind = self.pool(out, name='pool3')

            out = self.conv_bn_relu(pool3, 512, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='conv4_1')
            out = self.conv_bn_relu(out, 512, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='conv4_2')
            pool4, pool4_ind = self.pool(out, name='pool4')

            """
            Bottleneck
            """
            out = self.conv_bn_relu(pool4, 512, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='conv5_1')
            out = self.conv_bn_relu(out, 512, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='conv5_2')

            """
            Decoder
            """
            out = self.unpool(out, pool4_ind, name='unpool4')
            out = self.conv_bn_relu(out, 512, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='upconv4_2')
            out = self.conv_bn_relu(out, 256, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='upconv4_1')

            out = self.unpool(out, pool3_ind, name='unpool3')
            out = self.conv_bn_relu(out, 256, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='upconv3_2')
            out = self.conv_bn_relu(out, 128, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='upconv3_1')

            out = self.unpool(out, pool2_ind, name='unpool2')
            out = self.conv_bn_relu(out, 128, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='upconv2_2')
            out = self.conv_bn_relu(out, 64, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='upconv2_1')

            out = self.unpool(out, pool1_ind, name='unpool1')
            # Skipped connection
            out = tf.concat([out, conv1], axis=-1, name='skipped')
            # ------------------
            out = self.conv_bn_relu(out, 64, self.is_training, self.config.use_batch_norm, self.config.use_activation, kernel_size=(1, 1) name='upconv1_2')
            out = self.conv_bn_relu(out, 64, self.is_training, self.config.use_batch_norm, self.config.use_activation, name='upconv1_1')

            self.out = self.conv_predictor(out, use_activation=False)

            tf.add_to_collection('out', self.out)

        """
        Some operators for the training process
        """
        with tf.variable_scope('predictions'):
            self.predictions = tf.nn.sigmoid(self.out, name='pred')
            tf.add_to_collection('predictions', self.predictions)

        with tf.variable_scope('metrics'):
            self.loss = mt.dice_loss(y_true=self.y, y_pred=self.predictions)
            self.dice = mt.dice_coef(y_true=self.y, y_pred=self.predictions)
            self.iou  = mt.mean_iou(y_true=self.y, y_pred=self.predictions)

        with tf.variable_scope('train_step'):
            if self.config.optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.optimizer == 'Momentum':
                self.optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self.config.learning_rate,
                    momentum=self.config.momentum
                )
            else:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)

            if self.config.use_batch_norm:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
            else:
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.dice)
        tf.add_to_collection('train', self.iou)

    def init_saver(self):
        """
        Initialize the tensorflow saver that will be used in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)

    def conv_bn_relu(self, input_t, n_filters, is_training, use_batch_norm=True, use_activation=True, kernel_size=(3, 3), name=None):
        """
        Implementation of block Convolution -> Batch Normalization -> RELU.
        
        :param: input_t: Input Tensor with imensions [bs, h, w, n < n_filters].
        :param: n_filters: Number of filters to have in output Tensor.
        :param: is_training: tf.placeholder boolean variable.
        :param: use_batch_norm: True. True if you need to use Batch Normalization.
        :param: use_activation: True. True if you need to apply RELU function to the output.
        :param: kernel_size: (3, 3). Kernel size for the convolution operation.
        :param: name: None. Name of the scope.

        :return: Tensor with shape [bs, h, w, n_filters].
        """
        with tf.variable_scope(name) as scope:

            x = tf.layers.conv2d(
                input_t,
                filters=n_filters,
                kernel_size=kernel_size,
                padding="SAME",
                name=scope.name)

            if use_batch_norm:
                x = tf.layers.batch_normalization(
                    x,
                    training=is_training,
                    name=scope.name + '.bn'
                )

            if use_activation:
                x = tf.nn.relu(x)

            if self.config.debug:
                print(scope.name, x.shape)

        return x

    def pool(self, input_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name=None):
        """
        Pooling block.

        :param: input_t: Tensor with the shape [bs, h, w, f].
        :param: ksize: List with kernel sizes for every axis.
        :param: strides: Strides for the pooling operation.
        :param: name: Name of the scope.

        :return: value: Resulting Tensor of the pooling operation with size [bs, h/2, w/2, f].
        :return: index: Tensor containing indices for maximal values.
        """
        with tf.variable_scope(name) as scope:
            value, index = tf.nn.max_pool_with_argmax(
                input_t,
                ksize=ksize,
                strides=strides,
                padding='SAME',
                name=scope.name
            )
            if self.config.debug:
                print(scope.name, value.shape, index.shape)
        return value, index

    def unpool(self, pool, ind, ksize=[1, 2, 2, 1], name=None):
        """

        """
        with tf.variable_scope(name) as scope:
            input_shape = tf.shape(pool)
            output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

            flat_input_size = tf.cumprod(input_shape)[-1]
            flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

            pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
            batch_range = tf.reshape(
                tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                shape=tf.stack([input_shape[0], 1, 1, 1]))
            b = tf.ones_like(ind) * batch_range
            b = tf.reshape(b, tf.stack([flat_input_size, 1]))
            ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
            ind_ = tf.concat([b, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, tf.stack(output_shape))

            set_input_shape = pool.get_shape()
            set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
            ret.set_shape(set_output_shape)
        return ret

    def unpool_single(self, pool, name=None):
        """
        N-dimensional version of the unpooling operation from
        https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
        :param pool: A Tensor of shape [b, d0, d1, ..., dn, ch]
        :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
        """
        with tf.name_scope(name) as scope:
            sh = pool.get_shape().as_list()
            dim = len(sh[1:-1])
            out = (tf.reshape(pool, [-1] + sh[-dim:]))
            for i in range(dim, 0, -1):
                out = tf.concat([out, tf.zeros_like(out)], i)
            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
            out = tf.reshape(out, out_size, name=scope)

        if self.config.debug:
            print(scope.name, out.shape)
        return out

    def conv_predictor(self, input_t, filters=1, use_activation=True, activation=tf.nn.relu):
        with tf.variable_scope('out') as scope:
            x = tf.layers.conv2d(
                inputs=input_t,
                filters=filters,
                kernel_size=2,
                padding='SAME',
                name='conv_pred'
            )

            if use_activation:
                x = activation(x)

            if self.config.debug:
                print(scope.name, x.shape)

        return x
