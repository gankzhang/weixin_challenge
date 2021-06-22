# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import training_util

from tensorflow_estimator.python.estimator.canned import linear,dnn
from tensorflow import feature_column as fc
from base_comm import ACTION_LIST, STAGE_END_DAY, FEA_COLUMN_LIST
from base_evaluation import uAUC, compute_weighted_score
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_checkpoint_dir', './data/model', 'model dir')
flags.DEFINE_string('root_path', '../data/', 'data dir')
flags.DEFINE_integer('batch_size', 256, 'batch_size')
flags.DEFINE_integer('embed_dim', 10, 'embed_dim')
flags.DEFINE_float('learning_rate', 0.1, 'learning_rate')
flags.DEFINE_float('embed_l2', None, 'embedding l2 reg')
flags.DEFINE_integer('num_experts', 5, 'experts number')
flags.DEFINE_integer('num_epochs', 3, 'experts number')
flags.DEFINE_integer('use_feed_embedding', 0, 'if use feed embedding')

SEED = 2021

class MMOE(object):

    def __init__(self, linear_feature_columns, dnn_feature_columns, stage):
        """
        :param linear_feature_columns: List of tensorflow feature_column
        :param dnn_feature_columns: List of tensorflow feature_column
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        """
        super(MMOE, self).__init__()
        self.action_weight = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 2,
                                "comment": 1, "follow": 1}
        self.estimator = None
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.stage = stage
        tf.logging.set_verbosity(tf.logging.INFO)


    def build_estimator(self):
        if self.stage in ["evaluate", "offline_train"]:
            stage = "offline_train"
        else:
            stage = "online_train"
        model_checkpoint_stage_dir = os.path.join(FLAGS.model_checkpoint_dir, stage, 'mmoe')
        if not os.path.exists(model_checkpoint_stage_dir):
            # 如果模型目录不存在，则创建该目录
            os.makedirs(model_checkpoint_stage_dir)
        elif self.stage in ["online_train", "offline_train"]:
            # 训练时如果模型目录已存在，则清空目录
            del_file(model_checkpoint_stage_dir)
            pass
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1)
        config = tf.estimator.RunConfig(model_dir=model_checkpoint_stage_dir, tf_random_seed=SEED, log_step_count_steps=300)
        self.estimator = tf.estimator.Estimator(model_fn=self.model_fn, params = {'model_dir': model_checkpoint_stage_dir,
                                                                'linear_feature_columns': self.linear_feature_columns,
                                                                'dnn_feature_columns': self.dnn_feature_columns,
                                                                'dnn_hidden_units': [32, 8],
                                                                'dnn_optimizer': optimizer,
                                                                 'actions': 4,
                                                                 'experts':FLAGS.num_experts},
                                                                config = config)

    def model_fn(self,
                 features, # This is batch_features from input_fn
                 labels,   # This is batch_labels from input_fn
                 mode,     # An instance of tf.estimator.ModeKeys
                 params):
        # Use `input_layer` to apply the feature columns.
        with tf.variable_scope('dnn') as scope:
            if FLAGS.use_feed_embedding:
                feed_embedding = tf.nn.embedding_lookup(self.feed_embedding, features['feedid'])
            dnn_scope = scope.name
            dnn_logits_experts = list()
            dnn_embed = tf.feature_column.input_layer(features, params['dnn_feature_columns'])
            for expert_i in range(params['experts']):
                dnn_net = dnn_embed
                #shape of dnn_net (50)
                for unit in params['dnn_hidden_units']:
                    dnn_net = tf.layers.dense(dnn_net, units=unit, activation=tf.nn.relu)
                if FLAGS.use_feed_embedding:
                    dnn_embedding = tf.layers.dense(feed_embedding, 8, activation=tf.nn.relu)
                    dnn_net = tf.concat([tf.to_float(dnn_embedding),dnn_net], -1)
                dnn_logits_expert = tf.layers.dense(dnn_net, 1, activation=None)
                dnn_logits_experts.append(dnn_logits_expert)
            dnn_logits = tf.concat(dnn_logits_experts, -1)

            Use_attention_gate = True
            if Use_attention_gate:
                gates_list = list()
                for action_i in range(params['actions']):
                    gates = tf.layers.dense(dnn_embed, params['experts'])
                    gates = tf.expand_dims(gates,-1)
                    gates_list.append(gates)
                gates = tf.concat(gates_list, -1)
                gates = tf.nn.softmax(gates, 1)
            else:
                gates = tf.Variable(np.zeros((params['experts'], params['actions'])),trainable=True)
                gates = tf.nn.softmax(gates, 0)

        linear_logits_experts = list()
        linear_scopes = list()
        for expert_i in range(params['experts']):
            with tf.variable_scope(str(expert_i)+'_linear') as scope:
                linear_scopes.append(scope.name)
                #shape of linear_net (16)
                logit_fn = linear.linear_logit_fn_builder(
                    units=1,
                    feature_columns=params['linear_feature_columns'],
                    sparse_combiner='sum')
                linear_logits_expert = logit_fn(features=features)
                linear_logits_experts.append(linear_logits_expert)

        linear_logits = tf.concat(linear_logits_experts, -1)

        if Use_attention_gate:
            logits = tf.sigmoid(tf.matmul(tf.expand_dims(dnn_logits + linear_logits, 1), tf.to_float(gates)))
            logits = tf.reduce_sum(logits, 1)
        else:
            logits = tf.sigmoid(tf.matmul(dnn_logits + linear_logits, tf.to_float(gates)))

        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=logits)
        else:
            weights = tf.constant([[1.3], [1.1], [0.8], [0.8]])
            loss = -tf.reduce_sum(
                tf.matmul(
                tf.multiply(tf.to_float(labels), tf.log(logits+1e-8))
                + tf.multiply(1.0 - tf.to_float(labels), tf.log(1.0 - logits+1e-8))
                ,weights)
            )
            optimizer = params['dnn_optimizer']

            train_ops = list()
            variables_list = tf.all_variables()
            linear_optimizer = tf.train.FtrlOptimizer(0.005)
            for scope in linear_scopes:
                train_ops.append(linear_optimizer.minimize(
                    loss,var_list=ops.get_collection(
                        ops.GraphKeys.TRAINABLE_VARIABLES,
                        scope=scope)))
                for var in ops.get_collection(
                    ops.GraphKeys.TRAINABLE_VARIABLES,
                    scope=scope):
                    variables_list.remove(var)
            train_op = optimizer.minimize(
                loss, var_list=ops.get_collection(
                        ops.GraphKeys.TRAINABLE_VARIABLES,
                        scope=dnn_scope))
            train_ops.append(train_op)
            train_op = control_flow_ops.group(train_ops)
            global_step = training_util.get_global_step()
            with ops.control_dependencies([train_op]):
                train_op = state_ops.assign_add(global_step, 1).op

            metrics = dict()
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics)
        return spec


    def df_to_dataset(self, df, stage, shuffle=True, batch_size=128, num_epochs=1):
        '''
        把DataFrame转为tensorflow dataset
        :param df: pandas dataframe.
        :param stage: String. Including "online_train"/"offline_train"/"evaluate"/"submit"
        :param shuffle: Boolean.
        :param batch_size: Int. Size of each batch
        :param num_epochs: Int. Epochs num
        :return: tf.data.Dataset object.
        '''
        print(df.shape)
        print(df.columns)
        print("batch_size: ", batch_size)
        print("num_epochs: ", num_epochs)
        if FLAGS.use_feed_embedding:
            feed_embedding = pd.read_csv(os.path.join(FLAGS.root_path,'wechat_algo_data1', 'feed_embeddings.csv'))
            feed_embedding = dict(feed_embedding)
            new_feed_embedding = np.zeros((max(feed_embedding['feedid'])+1, 512))
            for id in range(len(feed_embedding)):
                embedding = feed_embedding['feed_embedding'][id]
                embedding = embedding.split(' ')
                embedding = [float(x) for x in embedding[:512]]
                new_feed_embedding[int(feed_embedding['feedid'][id])] = embedding
            feed_embedding = new_feed_embedding
            self.feed_embedding = tf.convert_to_tensor(feed_embedding)

        if stage != "submit":
            label = df[ACTION_LIST]
            ds = tf.data.Dataset.from_tensor_slices((dict(df), label))
        else:
            ds = tf.data.Dataset.from_tensor_slices((dict(df)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=SEED)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=batch_size * 10)
        if stage in ["online_train", "offline_train"]:
            ds = ds.repeat(num_epochs)
        return ds
    def input_fn_train(self, df, stage, num_epochs):
        return self.df_to_dataset(df, stage, shuffle=True, batch_size=FLAGS.batch_size,
                                  num_epochs=num_epochs)
    def input_fn_predict(self, df, stage):
        return self.df_to_dataset(df, stage, shuffle=False, batch_size=FLAGS.batch_size, num_epochs=1)

    def train(self, num_epochs = 1):
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=self.stage, action='all',
                                                                   day=STAGE_END_DAY[self.stage])
        stage_dir = os.path.join(FLAGS.root_path, self.stage, file_name)
        df = pd.read_csv(stage_dir)
        self.estimator.train(
            input_fn=lambda: self.input_fn_train(df, self.stage, num_epochs)
        )
    def evaluate(self, stage = 'evaluate'):
        """
        评估单个行为的uAUC值
        """
        action = "all"
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage=stage, action=action,
                                                                       day=STAGE_END_DAY[stage])
        evaluate_dir = os.path.join(FLAGS.root_path, stage, file_name)
        df = pd.read_csv(evaluate_dir)
        userid_list = df['userid'].astype(str).tolist()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, stage)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        score_dict = dict()
        predict_dict = dict()
        for i,action in enumerate(ACTION_LIST):
            labels = df[action].values
            uauc = uAUC(labels, predicts_df.values[:, i], userid_list)
            print(action,uauc)
            score_dict[action] = uauc
            predict_dict[action] = predicts_df.values[:, i]
        return df[["userid", "feedid"]], predict_dict, score_dict

    def predict(self):
        '''
        预测单个行为的发生概率
        '''
        file_name = "{stage}_{action}_{day}_concate_sample.csv".format(stage='submit', action="all",
                                                                       day=STAGE_END_DAY['submit'])
        submit_dir = os.path.join(FLAGS.root_path, 'submit', file_name)
        df = pd.read_csv(submit_dir)
        t = time.time()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(df, 'submit')
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        predict_dict = dict()
        for i, action in enumerate(ACTION_LIST):
            predict_dict[action] = predicts_df.values[:, i]
        # 计算2000条样本平均预测耗时（毫秒）
        ts = (time.time() - t) * 1000.0 / len(df) * 2000.0
        return df[["userid", "feedid"]], predict_dict, ts


def del_file(path):
    '''
    删除path目录下的所有内容
    '''
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            print("del: ", c_path)
            os.remove(c_path)

def get_feature_columns():
    '''
    获取特征列
    '''
    dnn_feature_columns = list()
    linear_feature_columns = list()
    # DNN features
    user_cate = fc.categorical_column_with_hash_bucket("userid", 40000, tf.int64)
    feed_cate = fc.categorical_column_with_hash_bucket("feedid", 240000, tf.int64)
    author_cate = fc.categorical_column_with_hash_bucket("authorid", 40000, tf.int64)
    bgm_singer_cate = fc.categorical_column_with_hash_bucket("bgm_singer_id", 40000, tf.int64)
    bgm_song_cate = fc.categorical_column_with_hash_bucket("bgm_song_id", 60000, tf.int64)
    user_embedding = fc.embedding_column(user_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    feed_embedding = fc.embedding_column(feed_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    author_embedding = fc.embedding_column(author_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    bgm_singer_embedding = fc.embedding_column(bgm_singer_cate, FLAGS.embed_dim)
    bgm_song_embedding = fc.embedding_column(bgm_song_cate, FLAGS.embed_dim)
    dnn_feature_columns.append(user_embedding)
    dnn_feature_columns.append(feed_embedding)
    dnn_feature_columns.append(author_embedding)
    dnn_feature_columns.append(bgm_singer_embedding)
    dnn_feature_columns.append(bgm_song_embedding)
    # Linear features
    video_seconds = fc.numeric_column("videoplayseconds", default_value=0.0)
    device = fc.numeric_column("device", default_value=0.0)
    linear_feature_columns.append(video_seconds)
    linear_feature_columns.append(device)
    # 行为统计特征
    for b in FEA_COLUMN_LIST:
        feed_b = fc.numeric_column(b + "sum", default_value=0.0)
        linear_feature_columns.append(feed_b)
        user_b = fc.numeric_column(b + "sum_user", default_value=0.0)
        linear_feature_columns.append(user_b)
    return dnn_feature_columns, linear_feature_columns


def main(argv):
    t = time.time()
    dnn_feature_columns, linear_feature_columns = get_feature_columns()
    stage = argv[1]
    print('Stage: %s' % stage)
    eval_dict = {}
    predict_dict = {}
    predict_time_cost = {}
    ids = None
    model = MMOE(linear_feature_columns, dnn_feature_columns, stage)
    model.build_estimator()

    if stage in ["online_train", "offline_train"]:
        # 训练 并评估
        for epoch_i in range(FLAGS.num_epochs):
            model.train(1)
            if stage == 'offline_train':
                weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                               "comment": 1, "follow": 1}
                ids, _, action_uauc = model.evaluate(stage)
                weight_auc = compute_weighted_score(action_uauc, weight_dict)
                print("Train Weighted uAUC: ", weight_auc)
                ids, _, action_uauc = model.evaluate()
                eval_dict = action_uauc
                weight_auc = compute_weighted_score(eval_dict, weight_dict)
                print("Evaluate Weighted uAUC: ", weight_auc)

        if stage == 'online_train':
            ids, logits, ts = model.predict()
            predict_dict = logits

    elif stage == "evaluate":
        # 评估线下测试集结果，计算单个行为的uAUC值，并保存预测结果
        ids, _, action_uauc = model.evaluate()
        eval_dict = action_uauc

    elif stage == "submit":
        # 预测线上测试集结果，保存预测结果
        ids, logits, ts = model.predict()
        predict_time_cost = ts
        predict_dict = logits

    if stage in ["evaluate", "offline_train", "online_train"]:
        # 计算所有行为的加权uAUC
        print(eval_dict)
        weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                       "comment": 1, "follow": 1}
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)

    # if stage in ["evaluate", "submit"]:
        # 保存所有行为的预测结果，生成submit文件
    actions = pd.DataFrame.from_dict(predict_dict)
    print("Actions:", actions)
    ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
    res = pd.concat([ids, actions], sort=False, axis=1)
    # 写文件
    file_name = "submit_" + str(int(time.time())) + ".csv"
    submit_file = os.path.join(FLAGS.root_path, stage, file_name)
    print('Save to: %s' % submit_file)
    res.to_csv(submit_file, index=False)

    if stage == "submit":
        print('不同目标行为2000条样本平均预测耗时（毫秒）：')
        print(predict_time_cost)
        print('单个目标行为2000条样本平均预测耗时（毫秒）：')
        print(np.mean([v for v in predict_time_cost.values()]))
    print('Time cost: %.2f s' % (time.time() - t))


if __name__ == "__main__":
    tf.app.run(main)
