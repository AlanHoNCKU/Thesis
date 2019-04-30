# -*- coding: UTF-8 -*-
"""
Date:2019/04/29
Author:Chia Yu,Ho
"""
# import setup_path 
import time
import airsim
import matplotlib.pyplot as plt
import pickle # 存取list

import numpy as np
import os
import tempfile
import pprint
import random

from q_learning import QLearningTable

def cnt_to_simulator():
    # 連線到 AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client

def drone_init(client):
    client.reset()
    cnt_to_simulator()
    client.takeoffAsync().join()
    client.moveByVelocityAsync(vx=random.randint(0,3), vy=random.randint(0,3), vz=-random.randint(0,3), duration=0.5).join()
    client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1).join()

def drone_action(client, actions):
    velocity = 2
    duration = 0.5
    if actions==0:#up
        client.moveByVelocityAsync(vx=0, vy=0, vz=-velocity, duration=duration).join()
        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=duration).join()
    if actions==1:#down
        client.moveByVelocityAsync(vx=0, vy=0, vz=velocity, duration=duration).join()
        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=duration).join()
    if actions==2:#left
        client.moveByVelocityAsync(vx=0, vy=velocity, vz=0, duration=duration).join()
        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=duration).join()
    if actions==3:#right
        client.moveByVelocityAsync(vx=0, vy=-velocity, vz=0, duration=duration).join()
        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=duration).join()
    if actions==4:#forward
        client.moveByVelocityAsync(vx=velocity, vy=0, vz=0, duration=duration).join()
        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=duration).join()
    if actions==5:#backward
        client.moveByVelocityAsync(vx=-velocity, vy=0, vz=0, duration=duration).join()
        client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=duration).join()

# 取得POS位置
def get_pos(client):
    x_val = client.simGetGroundTruthKinematics().position.x_val
    y_val = client.simGetGroundTruthKinematics().position.y_val
    z_val = client.simGetGroundTruthKinematics().position.z_val
    pos = np.array([x_val, y_val, z_val])
    return pos

def get_observation(now_pos):
    origin_pos = np.array([0,0,0])
    observation = np.linalg.norm(state - origin_pos)
    observation = int(observation)
    return observation


def drone_state(client):
    state = client.getMultirotorState()
    state_zval = state.kinematics_estimated.position.z_val
    state_zval = abs(state_zval)
    state_zval = round(state_zval,1)
    # state_zval = int(state_zval)
    return state_zval

def learning(client):
    success_cnt = 0
    success_cnt_history = []
    for episode in range(40):    
        drone_init(client)
        act_time = 0
        observation = drone_state(client)
        while True:
            act_time = act_time +1
            actions = RL.choose_action(str(int(observation)))
            # print('actions:' + str(actions))
            drone_action(client, actions)
            observation_ = drone_state(client)
            print('observation:' + str(observation),str(observation_))
            reward = 0
            if observation_ < observation:
                reward = reward + (1/(observation**2)+0.000001)
            # print('reward = ' + str(reward))
            # 任務成功
            if int(round(observation, 1)) == 0:
                reward = reward + 5
                
            print('reward = ' + str(reward))
            RL.learn(str(int(observation)), actions, reward, str(int(observation_)))
            act_time == act_time+1
            if act_time == 10 or observation == 0:
                observation_ = 'terminal'

            observation = observation_
            try:
                observation_detect = int(observation)
            except:
                pass
            if act_time == 10 or observation_detect == 0:
                if observation_detect == 0:
                    success_cnt = success_cnt + 1
                success_cnt_history.append(success_cnt)    
                print('成功次數'+str(success_cnt))
                print(success_cnt_history)
                break
    # drow_graph(success_cnt_history)
    print(success_cnt)


if __name__ == "__main__":
    client = cnt_to_simulator()
    action_space = ['rise', 'fall', 'left', 'right', 'forward', 'backward']
    n_actions = len(action_space)

    RL = QLearningTable(actions=list(range(n_actions)))
    # RL.read_qtable()
    learning(client)
    RL.save_qtable()
    print("over")