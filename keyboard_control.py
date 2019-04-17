# -*- coding: UTF-8 -*-
"""
Date:2019/04/12
Author:Chia Yu,Ho
"""
import airsim
import msvcrt

def connect():
    global client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

def action():
    exit_ = 0
    input_char = msvcrt.getch()
    if input_char == b'w':
        client.moveByVelocityAsync(5,0,0,1)
    elif input_char == b's':
        client.moveByVelocityAsync(-5,0,0,1)
    elif input_char == b'a':
        client.moveByVelocityAsync(0,-5,0,1)
    elif input_char == b'd':
        client.moveByVelocityAsync(0,5,0,1)
    elif input_char == b'k':
        client.moveByVelocityAsync(0,0,-5,1)
    elif input_char == b'j':
        client.moveByVelocityAsync(0,0,5,1)
    elif input_char == b'z':
        client.moveByVelocityAsync(0,0,0,1)
    elif input_char == b'l':
        client.landAsync()
    elif input_char == b'r':
        print(client.getPosition())
    elif input_char == b'o':
        exit_ = 1
        return exit_

if __name__ == '__main__':
    connect()
    client.takeoffAsync()
    while True:
        exit_ = action()
        if exit_ == 1:
            client.reset()
            break

