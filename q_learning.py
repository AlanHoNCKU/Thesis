# -*- coding: UTF-8 -*-
"""
Date:2019/04/29
Author:Chia Yu,Ho
"""

import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.7):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=float)

        # self.q_table = pd.DataFrame(dtype=float)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            # state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
            print(action)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        # print(self.q_table)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        print(self.q_table)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    def save_qtable(self):
        self.q_table.to_csv("train40.csv",index=True,sep=',')
    def read_qtable(self):
        self.q_table = pd.read_csv('train100.csv', index_col=0)
        self.q_table.columns = self.q_table.columns.map(int)
        self.q_table.index = self.q_table.index.map(str)
        print(self.q_table)