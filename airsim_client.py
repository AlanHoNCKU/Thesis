# -*- coding: UTF-8 -*-
"""
Date:2019/04/01
Author:Chia Yu,Ho
"""
import airsim
import os
import numpy as np
from PIL import Image

def init_drone_pos():
    global client
    client = airsim.MultirotorClient()
    client.reset()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    position = airsim.Vector3r(113.7,70.2,-0.5)
    heading = airsim.utils.to_quaternion(0, 0, 360)
    pose = airsim.Pose(position, heading)
    client.simSetVehiclePose(pose, True)

def get_img():
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
