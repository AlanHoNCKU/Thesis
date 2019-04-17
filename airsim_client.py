# -*- coding: UTF-8 -*-
"""
Date:2019/04/01
Author:Chia Yu,Ho
"""
import airsim
import os
import numpy as np
from PIL import Image

''' 有bug，直接移動後，飛行器無法正常飛行
def init_drone_pos(x, y, z ):
    global client
    client = airsim.MultirotorClient()
    client.reset()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    position = airsim.Vector3r(0 ,0 ,-10)
    position = airsim.Vector3r(x ,y ,z)
    heading = airsim.utils.to_quaternion(0, 0, 0)
    pose = airsim.Pose(position, heading)
    client.simSetVehiclePose(pose, True)
'''

# 連線
def connect():
    client = airsim.MultirotorClient()
    client.reset()
    # client.enableApiControl(False)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync()
    client.moveToPositionAsync(0, 0 , -56, 5).join()
    # client.moveByVelocityAsync(0,0,0,1)
    return client

# 初始化位置
def init_pos(client):
    client.moveByVelocityAsync(0,0,0,1).join()
    client.moveToPositionAsync(78.7, -34 , -56, 10).join()
    client.moveByVelocityAsync(0,0,0,1).join()

# 取得影像
def get_img(client):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    # get numpy array
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
    # reshape array to 4 channel image array H X W X 4
    img_rgba = img1d.reshape(response.height, response.width, 4)
    # 垂直翻轉圖像
    img_rgba = np.flipud(img_rgba)
    # 寫入成 png
    airsim.write_png(os.path.normpath('img_rgb' + '.png'), img_rgba)
    # rgba to rgb
    image = Image.open('img_rgb.png')
    image.load() # required for png.split()
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
    background.save('img_rgb.png', 'PNG', quality=100)
