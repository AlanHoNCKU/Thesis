# -*- coding: UTF-8 -*-
"""
Date:2019/05/01
Author:Chia Yu,Ho
"""
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, output_graph=False, e_greedy=0.7):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()


        if output_graph:
            tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # 全連接層1
        layer = tf.layers.dense(inputs=self.tf_obs, units=10, activation=tf.nn.tanh,
         kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
         bias_initializer=tf.constant_initializer(0,1), name='fc1')

        # 全連接層2
        all_act = tf.layers.dense(inputs=layer, units = self.n_actions, activation=None,
         kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
         bias_initializer=tf.constant_initializer(0,1), name='fc2')

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            #  neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
             neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
             loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        print(prob_weights)
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        print(action)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        np.seterr(divide='ignore',invalid='ignore')
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_model(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './train_model/papams20', write_meta_graph=False)
    
    def restore_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './train_model')