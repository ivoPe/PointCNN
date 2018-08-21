import tensorflow as tf
from pointcnn_cls import Net
import numpy as np
import pointfly as pf
from datetime import datetime
import os
from IPython.display import clear_output
from funcs_utils import get_batch


class Pcnn_classif:
    '''
    PointCNN class for classification.
    '''

    def __init__(self, setting):
        '''
        setting = Python module containing the config for the net, module
        '''
        # Sampling indices for prediction
        self.indi_pred = None
        # General settings
        self.setting = setting
        # Placeholders
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.indices = tf.placeholder(
            tf.int32, shape=(None, None, 2), name="indices")
        self.pts_fts = tf.placeholder(tf.float32, shape=(
            None, setting.cloud_point_nb, setting.data_dim), name='point_features')
        self.xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
        self.jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")

        if setting.regression:
            self.labels = tf.placeholder(
                tf.float32, shape=(None), name='labels')
            self.labels_re = tf.reshape(self.labels, (1, -1))
        else:
            self.labels = tf.placeholder(tf.int32, shape=(None), name='labels')
            self.one_hot_labels = tf.one_hot(
                indices=self.labels, depth=setting.num_class, dtype=tf.int32)

        # Net and logits
        self.pts_fts_sampled = tf.gather_nd(
            self.pts_fts, indices=self.indices, name='pts_fts_sampled')
        if setting.data_dim > 3:
            self.points_sampled, self.features_sampled = tf.split(
                self.pts_fts_sampled, [3, setting.data_dim - 3], axis=-1, name='split_points_features')
        else:
            self.points_sampled = self.pts_fts_sampled
            self.features_sampled = None
        if self.is_training == tf.constant(True):
            self.points_augmented = pf.augment(self.points_sampled, self.xforms, self.jitter_range)
        else:
            self.points_augmented = self.points_sampled
        self.net = Net(points=self.points_augmented, features=self.features_sampled,
                       is_training=self.is_training, setting=setting)
        self.logits = self.net.logits

        # Losses and Metrics
        if setting.regression:
            self.labels_tile = tf.tile(
                self.labels_re, (1, tf.shape(self.logits)[1]), name='labels_tile')
            self.labels_tile = tf.reshape(self.labels_tile, (tf.shape(
                self.logits)[0], tf.shape(self.logits)[1], -1))
            self.loss_op = tf.losses.mean_squared_error(
                self.labels_tile, self.logits)
            self.total_error = tf.reduce_sum(
                tf.square(tf.subtract(self.labels_tile, tf.reduce_mean(self.labels_tile))))
            self.unexplained_error = tf.reduce_sum(
                tf.square(tf.subtract(self.labels_tile, self.logits)))
            self.R_squared = tf.subtract(1., tf.div(self.unexplained_error, self.total_error))
            _ = tf.summary.scalar(
                'MSE', tensor=self.loss_op, collections=['train'])
            _ = tf.summary.scalar(
                'R2', tensor=self.R_squared, collections=['train'])
        else:
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
            self.mean_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.predictions, self.labe_accu), dtype=tf.float16))
            _ = tf.summary.scalar(
                'Accuracy', tensor=self.mean_accuracy, collections=['train'])
            _ = tf.summary.scalar(
                'Softmax_cross_entropy', tensor=self.loss_op, collections=['train'])
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
        summary_train_writer = tf.summary.FileWriter(
            summary_train_fold, sess.graph)
        # Decide if using Test set or not
        use_test = False
        if (X_test is not None) & (labels_test is not None):
            use_test = True
            if os.path.isdir(summary_test_fold) is False:
                os.mkdir(summary_test_fold)
            summary_test_writer = tf.summary.FileWriter(
                summary_test_fold, sess.graph)
            batch_id_test = get_batch(X_test, batch_size=batch_size,
                                      total_num_el=total_num_el)

        batch_id = get_batch(X_train, batch_size=batch_size,
                             total_num_el=total_num_el)
        all_prints = []
        for count, ba in enumerate(batch_id):
            val_x = X_train[ba]
            val_y = labels_train[ba]
            indi = pf.get_indices(
                val_x.shape[0], self.setting.sample_num, self.setting.cloud_point_nb, pool_setting=None)

            xforms_np, rotations_np = pf.get_xforms(batch_size,
                                                    rotation_range=self.setting.rotation_range,
                                                    scaling_range=self.setting.scaling_range,
                                                    order=self.setting.rotation_order)
            if self.setting.regression:
                _, loss_train = sess.run([self.train_op, self.loss_op], feed_dict={
                    self.pts_fts: val_x, self.labels: val_y, self.indices: indi, self.xforms: xforms_np,
                    self.jitter_range: np.array([self.setting.jitter]), self.is_training: True})
            else:
                _, loss_train, acc_train = sess.run([self.train_op, self.loss_op, self.mean_accuracy], feed_dict={
                    self.pts_fts: val_x, self.labels: val_y, self.indices: indi, self.xforms: xforms_np,
                    self.jitter_range: np.array([self.setting.jitter]), self.is_training: True})
            if count % summary_rate == 0:
                if use_test:
                    # Test summary
                    test_x = X_test[batch_id_test[count % len(batch_id_test)]]
                    test_y = labels_test[batch_id_test[count %
                                                       len(batch_id_test)]]
                    indi_test = pf.get_indices(
                        test_x.shape[0], self.setting.sample_num, self.setting.cloud_point_nb, pool_setting=None)
                    xforms_te, rotations_te = pf.get_xforms(batch_size,
                                                            rotation_range=self.setting.rotation_range_val,
                                                            scaling_range=self.setting.scaling_range_val,
                                                            order=self.setting.rotation_order)
                    if self.setting.regression:
                        loss_test, suma_test = sess.run([self.loss_op, self.summaries_op], feed_dict={
                            self.pts_fts: test_x, self.labels: test_y, self.indices: indi_test, self.xforms: xforms_te,
                            self.jitter_range: np.array([self.setting.jitter_val]), self.is_training: True})
                        # Train summary
                        suma = sess.run(self.summaries_op, feed_dict={
                            self.pts_fts: val_x, self.labels: val_y, self.indices: indi, self.xforms: xforms_np,
                            self.jitter_range: np.array([self.setting.jitter]), self.is_training: True})
                        new_print = '{}-[Val  ]-MSE train: {:.4f}  MSE test: {:.4f}'.format(
                            datetime.now(), loss_train, loss_test)
                    else:
                        acc_test, suma_test = sess.run([self.mean_accuracy, self.summaries_op], feed_dict={
                            self.pts_fts: test_x, self.labels: test_y, self.indices: indi_test, self.is_training: True})
                        # Train summary
                        suma = sess.run(self.summaries_op, feed_dict={
                            self.pts_fts: val_x, self.labels: val_y, self.indices: indi, self.xforms: xforms_np,
                            self.jitter_range: np.array([self.setting.jitter]), self.is_training: True})
                        new_print = '{}-[Val  ]-Loss: {:.4f}  Acc train: {:.4f}  Acc test: {:.4f}'.format(
                            datetime.now(), loss_train, acc_train, acc_test)
                    # Adding test summary
                    summary_test_writer.add_summary(suma_test, count)
                else:
                    if self.setting.regression:
                        suma = sess.run(self.summaries_op, feed_dict={
                            self.pts_fts: val_x, self.labels: val_y, self.indices: indi, self.xforms: xforms_np,
                            self.jitter_range: np.array([self.setting.jitter]), self.is_training: True})
                        new_print = '{}-[Val  ]-MSE train: {:.4f}'.format(
                            datetime.now(), loss_train)
                    else:
                        suma = sess.run(self.summaries_op, feed_dict={
                            self.pts_fts: val_x, self.labels: val_y, self.indices: indi, self.xforms: xforms_np,
                            self.jitter_range: np.array([self.setting.jitter]), self.is_training: True})
                        new_print = '{}-[Val  ]-Loss: {:.4f}  Acc train: {:.4f}'.format(
                            datetime.now(), loss_train, acc_train)

                summary_train_writer.add_summary(suma, count)

                all_prints.append(new_print)
                # Only print the last 5
                clear_output()
                all_prints = all_prints[-5:]
                for pri in all_prints:
                    print(pri)

    def predict(self, sess, X_test, batch_size=16):
        '''
        Predict labels and probas for X_test.
        ----
        Inputs:
        sess = Session with trained model, tf.Session
        X_test = Data test, numpy.array
        batch_size = Move through test by batch_size step, int
        ----
        Outputs:
        all_preds = Predicted labels, numpy.array
        all_probas = Predicted probas, numpy.array
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
            # We want to keep the same sampled points (idi_pred) for constant prediction results
            self._chek_indi_pred(val_x)
            if self.setting.regression:
                preds = sess.run(self.logits, feed_dict={
                    self.pts_fts: val_x, self.indices: self.indi_pred, self.is_training: False})
            else:
                preds, probas = sess.run([self.predictions, self.probs], feed_dict={
                    self.pts_fts: val_x, self.indices: self.indi_pred, self.is_training: False})
                all_probas.append(probas.reshape(
                    probas.shape[0], probas.shape[-1]))
            all_preds.append(preds.ravel())
        all_preds = np.concatenate(all_preds, axis=0)
        if self.setting.regression is False:
            all_probas = np.concatenate(all_probas, axis=0)

        return all_preds, all_probas

    def get_intermediate_ptfts(self, sess, one_cloud_point):
        '''
        Get the intermediate points for one cloud of points corresponding to
        one sample.
        ----
        Inputs:
        sess = Session with trained model, tf.Session
        one_cloud_point = One cloud of points, numpy.array
        ----
        Outputs:
        '''
        one_cloud = one_cloud_point.copy()
        # Shape check
        if len(one_cloud.shape) == 2:
            one_cloud = np.expand_dims(one_cloud, axis=0)
        # First check the indices to be sampled
        self._chek_indi_pred(one_cloud)
        # Defining the layers to get
        if self.setting.regression:
            proba = None
            logits, layer_points, layer_fts = sess.run(
                [self.logits, self.net.layer_pts, self.net.layer_fts],
                feed_dict={self.pts_fts: one_cloud,
                           self.indices: self.indi_pred, self.is_training: False})
        else:
            proba, logits, layer_points, layer_fts = sess.run(
                [self.probs, self.logits, self.net.layer_pts, self.net.layer_fts],
                feed_dict={self.pts_fts: one_cloud,
                           self.indices: self.indi_pred, self.is_training: False}
            )

        return proba, logits, layer_points, layer_fts

    def save_model(self, sess, save_path='models/model.ckpt'):
        '''
        Save the model using tensorlfow saver and ckpt files.
        ----
        Inputs:
        sess = Session used for training, tf.Session
        save_path = Path to the saved model, str
        ----
        Outputs:
        None
        '''
        saver = tf.train.Saver()
        saver.save(sess, save_path)

    def load_model(self, sess, save_path='models/model.ckpt'):
        '''
        Load the model given by save_path into the session sess.
        ! A class Pcnn_classif must have been __init__ !
        ----
        Inputs:
        sess = Session to restore model, tf.Session
        save_path = Path to the saved model, str
        ----
        Outputs:
        None
        '''
        saver = tf.train.Saver()
        saver.restore(sess, save_path)

    def _chek_indi_pred(self, val_x):
        '''
        Check if we need to change the sampled points.
        '''
        if self.indi_pred is None:
            indi = pf.get_indices(
                val_x.shape[0], self.setting.sample_num, self.setting.cloud_point_nb, pool_setting=None)
            self.indi_pred = indi
        elif self.indi_pred.shape[:-1] != val_x.shape[:-1]:
            indi = pf.get_indices(
                val_x.shape[0], self.setting.sample_num, self.setting.cloud_point_nb, pool_setting=None)
            self.indi_pred = indi
