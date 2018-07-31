import tensorflow as tf
from pointcnn_cls import Net
import numpy as np
import random
import pointfly as pf
from datetime import datetime
import os
from IPython.display import clear_output


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
        # Summaries
        self.summaries_op = tf.summary.merge_all('train')
        self.summaries_val_op = tf.summary.merge_all('val')
        # Finally add graph
        self.graph = tf.get_default_graph()

    def initialize(self):
        '''
        Initialize the variables and returns a session.
        '''
        sess = tf.Session(graph=tf.get_default_graph())
        sess.run(tf.global_variables_initializer())
        return sess

    def train(self, sess, X_train, labels_train, batch_size, total_num_el,
              X_test=None, labels_test=None,
              summary_folder='tf_summary', summary_rate=10):
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
        # Initialize the summary folder
        summary_train_fold = os.path.join(summary_folder, 'train')
        summary_test_fold = os.path.join(summary_folder, 'test')
        if os.path.isdir(summary_folder) is False:
            os.mkdir(summary_folder)
        if os.path.isdir(summary_train_fold) is False:
            os.mkdir(summary_train_fold)
        summary_train_writer = tf.summary.FileWriter(summary_train_fold, sess.graph)
        # Decide if using Test set or not
        use_test = False
        if (X_test is not None) & (labels_test is not None):
            use_test = True
            if os.path.isdir(summary_test_fold) is False:
                os.mkdir(summary_test_fold)
            summary_test_writer = tf.summary.FileWriter(summary_test_fold, sess.graph)
            batch_id_test = get_batch(X_test, batch_size=batch_size,
                                      total_num_el=total_num_el)

        train = [self.train_op, self.loss_op, self.mean_accuracy]
        batch_id = get_batch(X_train, batch_size=batch_size,
                             total_num_el=total_num_el)
        loss_arr = []
        acc_arr = []
        all_prints = []
        for count, ba in enumerate(batch_id):
            val_x = X_train[ba]
            val_y = labels_train[ba]
            indi = pf.get_indices(
                val_x.shape[0], self.setting.sample_num, self.setting.cloud_point_nb, pool_setting=None)
            _, loss_train, acc_train = sess.run(train, feed_dict={
                self.pts_fts: val_x, self.labels: val_y, self.indices: indi, self.is_training: True})
            if count % summary_rate == 0:
                if use_test:
                    # Test summary
                    test_x = X_test[batch_id_test[count % len(batch_id_test)]]
                    test_y = labels_test[batch_id_test[count % len(batch_id_test)]]
                    indi_test = pf.get_indices(
                        test_x.shape[0], self.setting.sample_num, self.setting.cloud_point_nb, pool_setting=None)
                    suma_test = sess.run(self.summaries_op, feed_dict={
                        self.pts_fts: test_x, self.labels: test_y, self.indices: indi_test, self.is_training: True})
                    summary_test_writer.add_summary(suma_test, count)
                    # Train summary
                    acc_test, suma = sess.run([self.mean_accuracy, self.summaries_op], feed_dict={
                        self.pts_fts: val_x, self.labels: val_y, self.indices: indi, self.is_training: True})
                    new_print = '{}-[Val  ]-Loss: {:.4f}  Acc train: {:.4f}  Acc test: {:.4f}'.format(
                        datetime.now(), loss_train, acc_train, acc_test)
                else:
                    suma = sess.run(self.summaries_op, feed_dict={
                        self.pts_fts: val_x, self.labels: val_y, self.indices: indi, self.is_training: True})
                    new_print = '{}-[Val  ]-Loss: {:.4f}  Acc train: {:.4f}'.format(
                        datetime.now(), loss_train, acc_train)

                summary_train_writer.add_summary(suma, count)

                all_prints.append(new_print)
                # Only print the last 5
                clear_output()
                all_prints = all_prints[-5:]
                for pri in all_prints:
                    print(pri)
            loss_arr.append([count, loss_train])
            acc_arr.append([count, acc_train])

        loss_arr = np.array(loss_arr)
        acc_arr = np.array(acc_arr)

        return loss_arr, acc_arr

    def predict(self, sess, X_test, batch_size=16):
        '''
        Predict labels for X_test.
        '''
        # Dim prepro
        if len(X_test.shape) == 2:
            test_val = np.expand_dims(X_test, axis=0)
        else:
            test_val = X_test

        pred_size = test_val.shape[0]
        all_preds = []
        all_probas = []
        for i in range(0, pred_size, batch_size):
            val_x = test_val[i:i + batch_size]
            indi = pf.get_indices(
                val_x.shape[0], self.setting.sample_num, self.setting.cloud_point_nb, pool_setting=None)
            preds, probas = sess.run([self.predictions, self.probs], feed_dict={
                self.pts_fts: val_x, self.indices: indi, self.is_training: False})
            all_preds.append(preds.ravel())
            all_probas.append(probas.reshape(probas.shape[0], probas.shape[-1]))
        all_preds = np.concatenate(all_preds, axis=0)
        all_probas = np.concatenate(all_probas, axis=0)

        return all_preds, all_probas

    def save_model(self, sess, save_folder='models'):
        '''
        Save the model using tensorlfow saver and ckpt files.
        '''
        # Creating the folder if does not exist
        if os.path.isdir(save_folder) is False:
            os.mkdir(save_folder)

        save_path = os.path.join(save_folder, 'model.ckpt')
        saver = tf.train.Saver()
        saver.save(sess, save_path)

    def load_model(self, sess, save_path='models/model.ckpt'):
        '''
        Load the model from the save folder.
        '''
        saver = tf.train.Saver()
        saver.restore(sess, save_path)


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
