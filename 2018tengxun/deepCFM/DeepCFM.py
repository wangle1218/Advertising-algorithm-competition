"""
Tensorflow implementation of DeepFM [1]

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


class DeepCFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,
                 num_multiVal_feat, sequence_length, vocab_size,
                 cnn_embedding_size=50,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu, 
                 fc_sizes=[512, 256],
                 filter_sizes=[[1,2,3,4,5],[3,4,5]], num_filters=50,
                 epoch=10, batch_size=4096,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.01, greater_is_better=True):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding
        self.cnn_embedding_size = cnn_embedding_size

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_multiVal_feat = num_multiVal_feat
        self.sequence_length = sequence_length
        self.fc_sizes = fc_sizes
        self.vocab_size = vocab_size

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="feat_value")  # None * F
            self.multiVal_feat = tf.placeholder(tf.int32, shape=[None, self.num_multiVal_feat, self.sequence_length],
                                                name='multiVal_feat') # None * N * L
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                             self.feat_index)  # None * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # ---------- first order term ----------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index) # None * F * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # None * F
            # self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0]) # None * F

            # ---------- second order term ---------------
            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            # self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # None * K

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size]) # None * (F*K)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer

            # ---------- CNN component ----------
            self.cnn_concat = []
            for i in range(self.num_multiVal_feat):
                embedded_chars = tf.nn.embedding_lookup(self.weights["cnn_embedding"], self.multiVal_feat[:,i,:])
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

                pooled_outputs = []
                for j, filter_size in enumerate(self.filter_sizes[0]):
                    with tf.name_scope("conv1-maxpool-%s-%s" % (j,filter_size)):
                        # Convolution Layer
                        filter_shape = [filter_size, self.cnn_embedding_size, 1, self.num_filters]
                        # 随机初始化卷积核的权重W
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                        conv = tf.nn.conv2d(
                            embedded_chars_expanded,      # input
                            W,                            # filter
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
                        if self.batch_norm:
                            conv = self.batch_norm_layer(conv, train_phase=self.train_phase, scope_bn="cnn1_bn_%s_%s" %(i,j))
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        # Maxpooling over the outputs
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                        pooled_outputs.append(pooled)

                # Combine all the pooled features
                num_filters_total = self.num_filters * len(self.filter_sizes[0])
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
                self.cnn_concat.append(h_pool_flat)

            self.cnn_concat = tf.transpose(self.cnn_concat, perm=(1,0,2))
            cnn_concat_expanded = tf.expand_dims(self.cnn_concat, -1)
            pooled_outputs = []
            for k, filter_size in enumerate(self.filter_sizes[1]):
                with tf.name_scope("conv2-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, num_filters_total, 1, self.num_filters]
                    # 随机初始化卷积核的权重W
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        cnn_concat_expanded,          # input
                        W,                            # filter
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    if self.batch_norm:
                        conv = self.batch_norm_layer(conv, train_phase=self.train_phase, scope_bn="cnn2_bn_%d" %k)
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.num_multiVal_feat - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
                    # print('pooled',pooled.get_shape())

            # Combine all the pooled features
            num_filters_total = self.num_filters * len(self.filter_sizes[1])
            h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            # print('self.h_pool_flat',self.h_pool_flat.get_shape())
            self.h_pool_flat = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_deep[1])

            # ---------- DeepFM ----------
            if self.use_fm and self.use_deep:
                self.concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep, self.h_pool_flat], axis=1)
            elif self.use_fm:
                self.concat_input = tf.concat([self.y_first_order, self.y_second_order, self.h_pool_flat], axis=1)
            elif self.use_deep:
                self.concat_input = tf.concat([self.y_deep, self.h_pool_flat],axis=1)

            print("self.concat_input",self.concat_input.get_shape())

            # ---------- full-connect ----------
            regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
            self.fc_layer = tf.layers.dense(self.concat_input, self.fc_sizes[0], activation=tf.nn.elu,\
                                        kernel_regularizer=regularizer, name="fc_layer%s" % 0)

            for i,fc_size in enumerate(self.fc_sizes[1:]):
                self.fc_layer = tf.layers.dense(self.fc_layer, fc_size, activation=tf.nn.elu,\
                                            kernel_regularizer=regularizer, name="fc_layer%s" % str(i+1))

            # self.out = tf.add(tf.matmul(self.fc_layer, self.weights["concat_projection"]), self.weights["concat_bias"])
            self.out = tf.layers.dense(self.fc_layer, 1, kernel_regularizer=regularizer, name="out")

            # loss
            if self.loss_type == "logloss":
                # self.out = tf.nn.sigmoid(self.out)
                # self.loss = tf.losses.log_loss(self.label, self.out)

                self.out=tf.sigmoid(self.out)
                logit_1=tf.log(self.out)
                logit_0=tf.log(1-self.out)
                self.loss=-tf.reduce_mean(self.label*logit_1+(1-self.label)*logit_0)
        
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters

            concat_num = self.concat_input.shape.as_list()[1]
            for fc_size in self.fc_sizes:
                total_parameters += concat_num * fc_size
                concat_num = fc_size
            total_parameters += self.fc_sizes[-1]
            if self.verbose > 0:
                print("#params: %d" % total_parameters)


    def _init_session(self):
        config = tf.ConfigProto(device_count={"cpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        weights["cnn_embedding"] = tf.Variable(
            tf.random_uniform([self.vocab_size-1, self.cnn_embedding_size], -1.0, 1.0),trainable=False,name="cnn_embedding") # vocab_size * embedding_size
        pad = tf.constant(0., shape=[1, self.cnn_embedding_size])
        weights["cnn_embedding"] = tf.concat([pad,weights["cnn_embedding"]], axis=0)

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        # if self.use_fm and self.use_deep:
        #     input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        # elif self.use_fm:
        #     input_size = self.field_size + self.embedding_size
        # elif self.use_deep:
        #     input_size = self.deep_layers[-1]
        # glorot = np.sqrt(2.0 / (self.fc_sizes[-1] + 1))
        # weights["concat_projection"] = tf.Variable(
        #                 np.random.normal(loc=0, scale=glorot, size=(self.fc_sizes[-1], 1)),
        #                 dtype=np.float32)  # layers[i-1]*layers[i]
        # weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    def get_batch(self, Xi, Xv, Xmv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], Xmv[start:end], [[y_] for y_ in y[start:end]]


    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c, d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)


    def fit_on_batch(self, Xi, Xv, Xmv ,y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.multiVal_feat: Xmv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def fit(self, Xi_train, Xv_train, Xmv_train, y_train,
            Xi_valid=None, Xv_valid=None, Xmv_valid=None ,y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t2 = time()
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, Xmv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, Xmv_batch, y_batch = self.get_batch(Xi_train, Xv_train,Xmv_train, y_train, self.batch_size, i)                
                loss = self.fit_on_batch(Xi_batch, Xv_batch,Xmv_batch, y_batch)
                if i % 50 == 49:
                    train_result = self.evaluate(Xi_batch, Xv_batch, Xmv_batch, y_batch)
                    print("[step %d] loss=%.5f  train-result=%.4f [%.1f s]"
                        % (i ,loss ,train_result, time() - t1))
                    t1 = time()
                if has_valid and i % 300 == 299:
                    valid_result = self.evaluate(Xi_valid, Xv_valid,Xmv_valid, y_valid)
                    print("[step %d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                        % (i , train_result, valid_result, time() - t1))

            train_result = self.evaluate(Xi_train, Xv_train,Xmv_train, y_train)
            valid_result = self.evaluate(Xi_valid, Xv_valid,Xmv_valid, y_valid)
            print("Epoch %s cost time=[%.1f s], train-auc=%.5f, valid auc=%.5f" % (epoch+1, time()-t2,train_result,valid_result))

        """
        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid
            for epoch in range(2):
                self.shuffle_in_unison_scary(Xi_train, Xv_train, Xmv_train,y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch,Xmv_batch, y_batch = self.get_batch(Xi_train, Xv_train,Xmv_train, y_train,
                                                                self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch,Xmv_batch, y_batch)
                # check
                train_result = self.evaluate(Xi_train, Xv_train,Xmv_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                    (self.greater_is_better and train_result > best_train_score) or \
                    ((not self.greater_is_better) and train_result < best_train_score):
                    break
        """

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False


    def predict(self, Xi, Xv, Xmv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, Xmv_batch, y_batch = self.get_batch(Xi, Xv, Xmv,dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.multiVal_feat: Xmv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, Xmv_batch, y_batch = self.get_batch(Xi, Xv, Xmv,dummy_y, self.batch_size, batch_index)

        return y_pred


    def evaluate(self, Xi, Xv, Xmv,y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv, Xmv)
        return self.eval_metric(y, y_pred)

