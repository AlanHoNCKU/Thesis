import time
import airsim
import matplotlib.pyplot as plt
import pickle # 存取list

import numpy as np
import os
import tempfile
import pprint
import random

import policy_gradient_v2 as PG


# 連線到 AirSim simulator
def cnt_to_simulator():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client

# 初始化 drone 狀態
def drone_init(client):
    client.reset()
    cnt_to_simulator()
    client.takeoffAsync().join()
    # client.moveByVelocityAsync(vx=random.randint(0,3), vy=random.randint(0,3), vz=-random.randint(0,3), duration=1).join()
    client.moveByVelocityAsync(vx=0, vy=0, vz=-5, duration=1).join()
    client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1).join()

# drone 的行為
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

# 取得當前位置
def get_pos(client):
    x_val = int(client.simGetGroundTruthKinematics().position.x_val)
    y_val = int(client.simGetGroundTruthKinematics().position.y_val)
    z_val = -int(client.simGetGroundTruthKinematics().position.z_val)
    pos = np.array([x_val, y_val, z_val])
    return pos

# 當前位置與原點的距離
def get_reward(now_pos):
    origin_pos = np.array([0,0,0])
    reward = np.linalg.norm(now_pos - origin_pos)
    reward = float(reward)
    return reward

if __name__ == "__main__":
    client = cnt_to_simulator()
    state_size = 3
    action_space = ['rise', 'fall', 'left', 'right', 'forward', 'backward']
    # action_size = len(action_space)
    action_size = 2

    agent = PG.REINFORCE(state_size, action_size)

    reward, episodes = [], []

    for e in range(100):
        act_time = 0
        done = False
        score = 0
        drone_init(client)
        state = np.reshape(get_pos(client), [1, state_size])

        while not done:
            act_time = act_time + 1
            action = agent.get_action(state)
            drone_action(client, action)
            next_state = np.reshape(get_pos(client), [1, state_size])
            now_reward = get_pos(client)[2]
            print(now_reward)
            if now_reward != 0:
                reward = 1/now_reward**10
                print(reward)
            # if get_reward(get_pos(client)) != 0:
            #     reward = 1/get_reward(get_pos(client))
            #     print(reward)

            agent.append_sample(state, action, reward)

            if act_time == 30:
                done = True
            if int(round(get_pos(client)[2], 1)) == 0:
                done = True
                reward = reward + 10
                reward = reward * 10

            score += reward
            state = next_state
            print(score)

            if done:
                agent.train_model()

        agent.model.save_weights("./save_model/reinforce_2.h5")
        # agent.model.load_weights("./save_model/reinforce_1.h5")