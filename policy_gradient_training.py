# -*- coding: UTF-8 -*-
"""
Date:2019/05/07
Author:Chia Yu,Ho
基於 Policy Gradient 訓練飛行器自主降落
"""
import airsim
import random
import numpy as np

from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt

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
    # client.moveByVelocityAsync(vx=random.randint(0,3), vy=random.randint(0,3), vz=-random.randint(0,3), duration=0.5).join()
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
    x_val = round(client.simGetGroundTruthKinematics().position.x_val, 2)
    y_val = round(client.simGetGroundTruthKinematics().position.y_val, 2)
    z_val = -round(client.simGetGroundTruthKinematics().position.z_val, 2)
    pos = np.array([x_val, y_val, z_val])
    return pos

# 當前位置與原點的距離
def get_dis(now_pos):
    origin_pos = np.array([0,0,0])
    dis = np.linalg.norm(now_pos - origin_pos)
    dis = float(dis)
    return dis

def get_plane_height_reward(now_pos):
    plane_pos = np.array([now_pos[0], now_pos[1]])
    print(plane_pos)
    origin_pos = np.array([0,0])
    plane_reward = np.linalg.norm(plane_pos - origin_pos)
    print(plane_reward)
    # plane_reward = float(reward)
    height_reward = -now_pos[2]
    print(plane_reward, height_reward)
    return plane_reward, height_reward

# 電子圍籬
def fence(now_pos):
    done = False
    if now_pos[0] < -3 or now_pos[0] > 3:
        done = True
    elif now_pos[1] < -3 or now_pos[1] > 3:
        done = True
    elif now_pos[2] > 7:
        done = True
    return done

if __name__ == "__main__":
    client = cnt_to_simulator()
    action_space = ['rise', 'fall', 'left', 'right', 'forward', 'backward']
    # n_actions = len(action_space)
    n_actions = 2
    RL = PolicyGradient(n_actions=n_actions, n_features=3, learning_rate=0.02, reward_decay=0.995)
    for i_episode in range(50):
        drone_init(client)
        observation = get_pos(client)
        done = 0
        dis = 0
        next_dis = 0
        while True:
            dis = get_dis(get_pos(client))
            action = RL.choose_action(observation)
            drone_action(client, action)
            observation_ = get_pos(client)
            next_dis = get_dis(get_pos(client))
            reward = dis - next_dis
            print(reward)

            if int(round(get_pos(client)[2], 1)) == 0:
                reward = reward + 10
                done = 1
            RL.store_transition(observation, action, reward)
            f_ = fence(get_pos(client))
            if f_ == True:
                done = True
                reward = reward -3
            
            if done:
                ep_rs_sum = sum(RL.ep_rs)
                print(ep_rs_sum)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                vt = RL.learn()
                break
            observation = observation_    

        

