import tensorflow as tf
from pointcnn_cls import Net
import numpy as np
import random
import pointfly as pf


class Pcnn_classif:
    '''
    PointCNN class for classification.
    '''

    def __init__(self, setting):
        self.setting = setting
        # Placeholders
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.indices = tf.placeholder(
            tf.int32, shape=(None, None, 2), name="indices")
        self.pts_fts = tf.placeholder(tf.float32, shape=(
            None, setting.cloud_point_nb, setting.data_dim), name='point_features')
        self.labels = tf.placeholder(tf.int32, shape=(None), name='labels')
        self.one_hot_labels = tf.one_hot(
            indices=self.labels, depth=setting.num_class, dtype=tf.int32)

        # Net and logits
        self.pts_fts_sampled = tf.gather_nd(
            self.pts_fts, indices=self.indices, name='pts_fts_sampled')
        self.points_sampled, self.features_sampled = tf.split(
            self.pts_fts_sampled, [3, setting.data_dim - 3], axis=-1, name='split_points_features')
        self.net = Net(points=self.points_sampled, features=self.features_sampled,
                       is_training=self.is_training, setting=setting)
        self.logits = self.net.logits

        # Losses and Metrics
        self.labels_tile = tf.tile(
            self.one_hot_labels, (1, tf.shape(self.logits)[1]), name='labels_tile')
        self.labels_tile = tf.reshape(self.labels_tile, (tf.shape(
            self.logits)[0], tf.shape(self.logits)[1], -1))
        self.probs = tf.nn.softmax(self.logits, name='probs')
        _, self.predictions = tf.nn.top_k(self.probs, name='predictions')
        self.labe_accu = tf.reshape(
            tf.argmax(self.labels_tile, axis=-1), shape=(tf.shape(self.predictions)))
        self.labe_accu = tf.cast(self.labe_accu, self.predictions.dtype)
        self.loss_op = tf.losses.softmax_cross_entropy(
            self.labels_tile, self.logits)
        _ = tf.summary.scalar(
            'train/loss', tensor=self.loss_op, collections=['train'])
        self.mean_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predictions, self.labe_accu), dtype=tf.float16))
        _ = tf.summary.scalar(
            'train/Accuracy', tensor=self.mean_accuracy, collections=['train'])
        _ = tf.summary.scalar(
            'val/Accuracy', tensor=self.mean_accuracy, collections=['val'])
        # Optimizer and training step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        lr_exp_op = tf.train.exponential_decay(
            setting.learning_rate_base, self.global_step, setting.decay_steps, setting.decay_rate, staircase=True)
        lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
        _ = tf.summary.scalar(
            'learning_rate', tensor=lr_clip_op, collections=['train'])
        reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
        if setting.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=lr_clip_op, epsilon=setting.epsilon)
        elif setting.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(
                self.loss_op + reg_loss, global_step=self.global_step)

        # Finally add graph
        self.graph = tf.get_default_graph()

    def initialize(self):
        '''
        Initialize the variables and returns a session.
        '''
        sess = tf.Session(graph=tf.get_default_graph())
        sess.run(tf.global_variables_initializer())
        return sess

    def train(self, sess, X_train, labels_train, batch_size, total_num_el):
        '''
        Train the Network for classification
        ----
        Inputs:
        sess = Session on which to run the graph, tensorflow.Session
        X_train = Points and features dataset, numpy.array
        labels_train = labels for classification, numpy.array
        batch_size = Batch size for train, int
        total_num_el = Total number of element to use for training, int
        '''
        train = [self.train_op, self.loss_op, self.mean_accuracy]
        batch_id = get_batch(X_train, batch_size=batch_size,
                             total_num_el=total_num_el)
        loss_arr = []
        acc_arr = []
        for count, ba in enumerate(batch_id):
            val_x = X_train[ba]
            val_y = labels_train[ba]
            indi = pf.get_indices(
                val_x.shape[0], self.setting.sample_num, self.setting.cloud_point_nb, pool_setting=None)
            _, loss_train, acc_train = sess.run(train, feed_dict={
                self.pts_fts: val_x, self.labels: val_y, self.indices: indi, self.is_training: True})
            loss_arr.append([count, loss_train])
            acc_arr.append([count, acc_train])
        loss_arr = np.array(loss_arr)
        acc_arr = np.array(acc_arr)

        return loss_arr, acc_arr

    def predict(self, sess, X_test):
        '''
        Predict labels for X_test.
        '''
        indi = pf.get_indices(
            X_test.shape[0], self.setting.sample_num, self.setting.cloud_point_nb, pool_setting=None)
        label_pred = sess.run([self.predictions, self.probs], feed_dict={
            self.pts_fts: X_test, self.indices: indi, self.is_training: False})

        return label_pred


def get_batch(dataset, batch_size, total_num_el):
    '''
    Get the indices for the batches for training
    '''
    ori_num_elements = len(dataset)
    inda = np.arange(ori_num_elements)
    all_batches = []

    batch_nb = int(total_num_el / batch_size)
    one_epoch_batch_nb = int(ori_num_elements / batch_size)
    iter_nb = max(int(batch_nb / one_epoch_batch_nb), 1)

    for i in range(iter_nb + 1):
        random.shuffle(inda)
        new_batches = [inda[u:u + batch_size]
                       for u in range(0, len(dataset), batch_size)]
        if new_batches[-1].shape[0] == batch_size:
            all_batches += new_batches
        else:
            all_batches += new_batches[:-1]

    return all_batches
